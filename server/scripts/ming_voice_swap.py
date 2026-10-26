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


def synthesize(base_url: str, text: str, voice: str, language: str, seed: int | None) -> tuple[int, np.ndarray]:
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
    response = requests.post(f"{base_url.rstrip('/')}/v1/audio/speech", json=payload, timeout=600)
    response.raise_for_status()
    audio, rate = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=True)
    return int(rate), audio.mean(axis=1)


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
    parser.add_argument("--retakes", type=int, default=0, help="re-synthesize a turn when its QA WER exceeds the threshold")
    parser.add_argument("--retake-threshold", type=float, default=0.04)
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
    if args.retakes > 0:
        from transcription import FasterWhisperTranscriber

        transcriber = FasterWhisperTranscriber()
        print(f"retakes: up to {args.retakes} per turn when WER > {args.retake_threshold:.0%}, seeds={seeds}")
    started = time.time()
    done = 0
    retake_count = 0

    for stem, voice in assignments:
        language = transcripts[stem].get("language") or args.language
        if len(str(language)) > 3:  # engine wants codes like "en"
            language = args.language
        for turn in turns[stem]:
            start = float(turn["start_seconds"])
            end = float(turn["end_seconds"])
            window = end - start
            text = str(turn.get("text", "")).strip()
            if window <= 0 or not text:
                continue

            best = None
            attempts = 0
            max_attempts = 1 + max(0, args.retakes)
            while attempts < max_attempts:
                seed = seeds[attempts % len(seeds)] if seeds else None
                source_rate, generated = synthesize(args.base_url, text, voice, language, seed)
                attempts += 1
                wer = None
                if transcriber is not None:
                    wer, _ = transcribe_errors(
                        transcriber, generated, source_rate, text, Path(args.out_dir or "."),
                        args.qa_model, args.qa_device, args.language,
                    )
                candidate = (wer if wer is not None else 0.0, generated, source_rate, seed, attempts)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                if wer is None or wer <= args.retake_threshold:
                    break
            if best[4] > 1:
                retake_count += 1

            natural = prepare_natural_audio(best[1], best[2], output_rate)
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
                    "strategy": "natural_speed_no_stretch",
                    "attempts": best[4],
                    "qa_wer": round(best[0], 4),
                    "seed": best[3],
                }
            )
            done += 1
            print(
                f"  [{done}] {stem:<8} src={window:5.2f}s gen={duration:5.2f}s "
                f"({duration / window:4.2f}x) took={best[4]}"
                + (f" wer={best[0] * 100:4.1f}%" if transcriber is not None else "")
                + f" {text[:52]!r}",
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
    print(f"mix: {mix_mp3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
