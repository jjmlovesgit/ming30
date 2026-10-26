"""Ming 3.0 voice swap over an existing SAM_fastapi run.

Reuses SAM_fastapi's own turn building, timeline scheduling and assembly, and
swaps only the TTS client (OpenAI-shaped HTTP) for the Ming adapter.

    python ming_voice_swap.py --run "<run dir>" --voice-one US_James --voice-two US_Emma
    python ming_voice_swap.py --run "<run dir>" --limit 6 --tag smoke
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r"C:\Projects\SAM_fastapi")
sys.path.insert(0, r"C:\Projects\ming30\server")

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

import tts_pipeline as tp
from tts_pipeline import (
    PocketTTSClient,
    add_to_timeline,
    apply_editorial_script,
    build_speaker_turns,
    build_tight_dialogue_schedule,
    build_timing_diagnostics,
    encode_mp3,
    finish_timeline,
    mix_speaker_tracks,
    prepare_natural_audio,
    resolve_cross_stem_leakage,
    write_pcm16_wav,
)


def synthesize(
    base_url: str,
    text: str,
    voice: str,
    language: str,
    seed: int | None,
    instruct: str | None = None,
) -> tuple[int, np.ndarray]:
    """One TTS request through the OpenAI-shaped adapter, optionally seed-pinned."""

    payload: dict = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "language": language,
    }
    if seed is not None:
        payload["options"] = {"seed": seed}
    if instruct:
        payload["instructions"] = instruct
    response = requests.post(f"{base_url.rstrip('/')}/v1/audio/speech", json=payload, timeout=600)
    response.raise_for_status()
    audio, rate = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=True)
    return int(rate), audio.mean(axis=1)


def rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(audio))) + 1e-9))


def load_source(run_dir: Path, stem: str, rate: int) -> np.ndarray | None:
    path = run_dir / f"{stem}.wav"
    if not path.exists():
        return None
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != rate:
        from scipy.signal import resample_poly

        divisor = np.gcd(int(sample_rate), int(rate))
        mono = resample_poly(mono, rate // divisor, int(sample_rate) // divisor).astype(np.float32)
    return mono


BACKCHANNEL = re.compile(
    r"^(right|right\?|yeah|yeah\.|yes|exactly|exactly\.|absolutely|sure|well|okay|"
    r"i mean|i see|how so\?|wrong\?|why\?|and\?|really\?|of course)[\s.,!?]*$",
    re.IGNORECASE,
)


def is_backchannel(text: str, max_words: int) -> bool:
    words = text.split()
    return len(words) <= max_words or bool(BACKCHANNEL.match(text.strip()))


def transcribe_errors(transcriber, audio: np.ndarray, rate: int, text: str, out_dir: Path, model: str, device: str, language: str) -> tuple[float, str]:
    """Word error rate of a generated take, transcribed by faster-whisper."""

    from transcription import word_match_metrics

    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(out_dir))
    path = Path(handle.name)
    handle.close()
    try:
        sf.write(path, np.clip(audio, -1.0, 1.0), rate)
        transcript = transcriber.transcribe_stem(path, "probe", "probe", model, device, language)
    finally:
        path.unlink(missing_ok=True)
    recognized = " ".join(segment["text"] for segment in transcript.get("segments", []))
    metrics = word_match_metrics(text, recognized)
    return float(metrics["word_error_rate"]), recognized


def load_run(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8070")
    parser.add_argument("--voice-one", default="US_James")
    parser.add_argument("--voice-two", default="US_Emma")
    parser.add_argument("--language", default="en")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--limit", type=int, default=0, help="synthesize only N turns per stem")
    parser.add_argument("--out-dir", default="", help="where to write outputs (default: the run directory)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--retakes", type=int, default=0, help="extra takes allowed when a take fails the WER gate")
    parser.add_argument("--retake-threshold", type=float, default=0.04)
    parser.add_argument("--candidates", type=int, default=1, help="passing takes to generate per turn before selecting")
    parser.add_argument("--backchannel-words", type=int, default=3, help="turns at or below this length get the hint")
    parser.add_argument("--backchannel-instruct", default="", help="delivery hint for short reactive turns")
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=0.0,
        help="weight of source-energy matching in take selection; off by default because per-stem "
        "timings do not reliably align to the stem WAVs after editorial reassignment",
    )
    parser.add_argument("--fit-pacing", action="store_true", help="clamp per-turn duration toward the source window")
    parser.add_argument("--fit-min", type=float, default=0.85, help="fastest allowed tempo factor")
    parser.add_argument("--fit-max", type=float, default=1.15, help="slowest allowed tempo factor")
    parser.add_argument("--seeds", default="1895,7,99,1234")
    parser.add_argument("--qa-model", default="large-v3-turbo")
    parser.add_argument("--qa-device", default="cuda")
    parser.add_argument("--strict-leakage", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    manifest = load_run(run_dir)

    tp.OUTPUT_SAMPLE_RATE = args.sample_rate
    output_rate = args.sample_rate

    stems = (manifest.get("transcription") or {}).get("stems") or {}
    transcripts = {}
    for stem in ("target", "residual"):
        filename = (stems.get(stem) or {}).get("json")
        if not filename:
            print(f"missing {stem} transcript in manifest", file=sys.stderr)
            return 2
        transcripts[stem] = json.loads((run_dir / filename).read_text(encoding="utf-8"))

    editorial = manifest.get("editorial_script")
    if not editorial:
        editorial = (manifest.get("voice_swap") or {}).get("editorial_script") or {}
    transcripts, forced_breaks = apply_editorial_script(transcripts, editorial)
    transcripts, leakage = resolve_cross_stem_leakage(transcripts)
    print(
        f"editorial: {sum(len(v) for v in editorial.values()) if editorial else 0} entries, "
        f"leakage conflicts={leakage['detected_conflicts']} "
        f"resolved={leakage['auto_resolved_conflicts']} unresolved={leakage['unresolved_conflicts']}"
    )
    if leakage["unresolved_conflicts"] and args.strict_leakage:
        print("unresolved cross-stem conflicts; rerun without --strict-leakage", file=sys.stderr)
        return 3

    turns = build_speaker_turns(transcripts, forced_breaks)
    if args.limit:
        turns = {stem: items[: args.limit] for stem, items in turns.items()}
    counts = {stem: len(items) for stem, items in turns.items()}
    print(f"turns: {counts} total={sum(counts.values())}")

    client = PocketTTSClient(args.base_url)
    voices = {item.get("voice_id") for item in client.list_voices()}
    for voice in (args.voice_one, args.voice_two):
        if voice not in voices:
            print(f"voice '{voice}' not offered by {args.base_url}: {sorted(v for v in voices if v)}", file=sys.stderr)
            return 4

    assignments = (
        ("target", args.voice_one),
        ("residual", args.voice_two),
    )
    generated_segments = {"target": [], "residual": []}
    timing_records = {"target": [], "residual": []}
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    transcriber = None
    if args.retakes > 0 or args.candidates > 1:
        from transcription import FasterWhisperTranscriber

        transcriber = FasterWhisperTranscriber()
        print(
            f"selection: {args.candidates} passing take(s), up to "
            f"{args.candidates + max(0, args.retakes)} takes per turn, "
            f"WER gate {args.retake_threshold:.0%}, seeds={seeds}"
        )
    started = time.time()
    done = 0
    retake_count = 0
    instructed = 0
    hint_disabled = False
    ratios: list[float] = []
    energy_deltas: list[float] = []

    for stem, voice in assignments:
        language = transcripts[stem].get("language") or args.language
        if len(str(language)) > 3:  # engine wants codes like "en"
            language = args.language
        source_audio = load_source(run_dir, stem, output_rate)
        if source_audio is None:
            print(f"  note: {stem}.wav not found; energy scoring disabled for this stem")
        for turn in turns[stem]:
            start = float(turn["start_seconds"])
            end = float(turn["end_seconds"])
            window = end - start
            text = str(turn.get("text", "")).strip()
            if window <= 0 or not text:
                continue

            hint = None
            if args.backchannel_instruct and not hint_disabled and is_backchannel(text, args.backchannel_words):
                hint = args.backchannel_instruct
            db_source = None
            if source_audio is not None:
                db_source = rms_db(source_audio[int(start * output_rate): int(end * output_rate)])

            takes = []
            attempt = 0
            passing = 0
            max_takes = args.candidates + max(0, args.retakes)
            while attempt < max_takes and passing < args.candidates:
                seed = seeds[attempt % len(seeds)] if seeds else None
                try:
                    source_rate, generated = synthesize(args.base_url, text, voice, language, seed, hint)
                except requests.HTTPError as exc:
                    detail = ""
                    try:
                        detail = exc.response.text
                    except Exception:  # noqa: BLE001 - best effort
                        detail = str(exc)
                    if hint and "instruct" in detail.lower() and not hint_disabled:
                        hint_disabled = True
                        print(
                            "  note: this engine rejects raw instructions for built-in voices "
                            "('Ming IP uses structured controls'); delivery hints disabled for the rest of the run"
                        )
                        hint = None
                        source_rate, generated = synthesize(args.base_url, text, voice, language, seed, None)
                    else:
                        raise
                attempt += 1
                duration = generated.size / source_rate
                wer = None
                if transcriber is not None:
                    wer, _ = transcribe_errors(
                        transcriber, generated, source_rate, text, Path(args.out_dir or "."),
                        args.qa_model, args.qa_device, args.language,
                    )
                ok = wer is None or wer <= args.retake_threshold
                passing += 1 if ok else 0
                ratio = duration / window
                db_take = rms_db(generated)
                energy = abs(db_take - db_source) / 6.0 if db_source is not None else 0.0
                takes.append(
                    {
                        "ok": ok,
                        "wer": wer,
                        "audio": generated,
                        "rate": source_rate,
                        "seed": seed,
                        "ratio": ratio,
                        "db": db_take,
                        "db_source": db_source,
                        "score": abs(ratio - 1.0) + args.energy_weight * min(energy, 3.0),
                        "instruct": hint,
                    }
                )
                if passing >= args.candidates:
                    break

            eligible = [take for take in takes if take["ok"]] or takes
            best = min(
                eligible,
                key=lambda take: (
                    take["score"] if take["ok"] else 1.0 + (take["wer"] or 0.0),
                    take["score"],
                ),
            )
            if len(takes) > 1:
                retake_count += 1
            if best["instruct"]:
                instructed += 1
            ratios.append(best["ratio"])
            if best["db_source"] is not None:
                energy_deltas.append(abs(best["db"] - best["db_source"]))

            natural = prepare_natural_audio(best["audio"], best["rate"], output_rate)
            speed = 1.0
            if args.fit_pacing and natural.size:
                speed = float(natural.size / output_rate) / window
                speed = min(max(speed, args.fit_min), args.fit_max)
                if abs(speed - 1.0) > 0.005:
                    from fractions import Fraction

                    fraction = Fraction(speed).limit_denominator(200)
                    natural = resample_poly(natural, fraction.denominator, fraction.numerator).astype(np.float32)
                else:
                    speed = 1.0
            duration = natural.size / output_rate
            generated_segments[stem].append((turn.get("id"), natural))
            timing_records[stem].append(
                {
                    "segment_id": turn.get("id"),
                    "text": text,
                    "stem": stem,
                    "voice": voice,
                    "boundary_reason": turn.get("boundary_reason"),
                    "start_seconds": start,
                    "transcript_end_seconds": end,
                    "transcript_window_seconds": round(window, 3),
                    "generated_seconds": round(duration, 3),
                    "natural_end_seconds": round(start + duration, 3),
                    "overrun_seconds": round(max(0.0, duration - window), 3),
                    "strategy": "natural_speed_no_stretch" if not args.fit_pacing else f"tempo_fit_{speed:.3f}",
                    "attempts": len(takes),
                    "qa_wer": round(best["wer"], 4) if best["wer"] is not None else None,
                    "seed": best["seed"],
                    "pacing_ratio": round(best["ratio"], 3),
                    "energy_delta_db": round(best["db"] - best["db_source"], 2) if best["db_source"] is not None else None,
                    "selection_score": round(best["score"], 3),
                    "instruct": best["instruct"],
                }
            )
            done += 1
            print(
                f"  [{done}] {stem:<8} src={window:5.2f}s gen={duration:5.2f}s "
                f"({duration / window:4.2f}x) takes={len(takes)}"
                + (f" wer={best['wer'] * 100:4.1f}%" if best["wer"] is not None else "")
                + (f" db={best['db'] - best['db_source']:+.1f}" if best["db_source"] is not None else "")
                + (" [hint]" if best["instruct"] else "")
                + f" {text[:44]!r}",
                flush=True,
            )

    synth_seconds = time.time() - started
    schedule = build_tight_dialogue_schedule(timing_records)
    placements = schedule["placements"]
    for stem, records in timing_records.items():
        for record in records:
            record.update(placements[(stem, record["segment_id"])])

    timeline_samples = max(1, round(schedule["output_duration_seconds"] * output_rate))
    tracks = {}
    diagnostics = {}
    for stem, _voice in assignments:
        timeline = np.zeros(timeline_samples, dtype=np.float32)
        weights = np.zeros(timeline_samples, dtype=np.float32)
        for turn_id, audio in generated_segments[stem]:
            rebuilt_start = placements[(stem, turn_id)]["rebuilt_start_seconds"]
            add_to_timeline(timeline, weights, audio, rebuilt_start, output_rate)
        tracks[stem] = finish_timeline(timeline, weights)
        diagnostics[stem] = build_timing_diagnostics(timing_records[stem])

    suffix = args.tag or datetime.now().strftime("%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    one = out_dir / f"speaker_1_swapped_ming_{suffix}.wav"
    two = out_dir / f"speaker_2_swapped_ming_{suffix}.wav"
    mix_wav = out_dir / f"voice_swapped_mix_ming_{suffix}.wav"
    mix_mp3 = out_dir / f"voice_swapped_mix_ming_{suffix}.mp3"
    write_pcm16_wav(one, tracks["target"])
    write_pcm16_wav(two, tracks["residual"])
    write_pcm16_wav(mix_wav, mix_speaker_tracks(tracks["target"], tracks["residual"]))
    encode_mp3(mix_wav, mix_mp3)

    source_duration = float((manifest.get("separation") or {}).get("output_duration_seconds") or 0.0)
    summary = {
        "engine": "ming30 (audio.cpp ming3_tts q8_0)",
        "api_url": args.base_url,
        "voices": {"speaker_1_target": args.voice_one, "speaker_2_residual": args.voice_two},
        "sample_rate": output_rate,
        "turn_counts": counts,
        "assembly_policy": schedule["policy"],
        "source_duration_seconds": round(source_duration, 3),
        "output_duration_seconds": round(timeline_samples / output_rate, 3),
        "synthesis_seconds": round(synth_seconds, 2),
        "retakes": retake_count,
        "retake_threshold": args.retake_threshold,
        "candidates": args.candidates,
        "instructed_turns": instructed,
        "backchannel_instruct": args.backchannel_instruct or None,
        "pacing": {
            "turns": len(ratios),
            "median_abs_ratio_error": round(float(np.median([abs(value - 1.0) for value in ratios])), 3) if ratios else None,
            "mean_ratio": round(float(np.mean(ratios)), 3) if ratios else None,
            "within_10_percent": sum(1 for value in ratios if abs(value - 1.0) <= 0.10),
        },
        "energy": {
            "median_abs_delta_db": round(float(np.median(energy_deltas)), 2) if energy_deltas else None,
            "stdev_delta_db": round(float(np.std(energy_deltas)), 2) if energy_deltas else None,
        },
        "outputs": {"mix_wav": mix_wav.name, "mix_mp3": mix_mp3.name, "speaker_one": one.name, "speaker_two": two.name},
        "timing_diagnostics": {
            stem: {
                "segments_needing_review": diag["segments_needing_review"],
                "transitions_needing_review": diag["transitions_needing_review"],
                "generated_speech_seconds": diag["generated_speech_seconds"],
            }
            for stem, diag in diagnostics.items()
        },
        "segment_timing": timing_records,
    }
    (out_dir / f"voice_swap_ming_{suffix}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\nsource={summary['source_duration_seconds']}s output={summary['output_duration_seconds']}s "
        f"synth={synth_seconds:.1f}s rate={output_rate}"
    )
    for stem, diag in diagnostics.items():
        print(f"  {stem}: generated={diag['generated_speech_seconds']}s review={diag['segments_needing_review']}/{diag['segment_count']}")
    if ratios:
        within = sum(1 for value in ratios if abs(value - 1.0) <= 0.10)
        print(
            f"  pacing: mean_ratio={np.mean(ratios):.3f} median|err|={np.median([abs(v - 1.0) for v in ratios]):.3f} "
            f"within10%={within}/{len(ratios)}"
        )
    if energy_deltas:
        print(f"  energy: median|delta|={np.median(energy_deltas):.2f} dB stdev={np.std(energy_deltas):.2f} dB")
    if instructed:
        print(f"  backchannel hints applied to {instructed} turn(s)")
    print(f"mix: {mix_mp3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
