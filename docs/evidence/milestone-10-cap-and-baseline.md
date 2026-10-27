# Duration cap documented; generation smoke baseline for the shared engine

## The cap, as a caller-facing parameter

`max_tokens` in the family spec is now described as what it actually is:

> **Duration cap**: maximum generated acoustic patches. Mirrors the checkpoint's
> `maximum_decode_steps` (default **250 patches = 20.0 s** at 12.5 patches/s). The
> checkpoint has no end-of-sentence decision (`old_shift` decode), so this is the only
> length control; shorter renders are explicit caps.

Default set to 250 to match the checkpoint. Our 30-patch (2.40 s) render is therefore a
deliberate cap, not the model's behaviour — as established in the previous note.

## Shared-engine regression is now a generation check, not just registration

`--list-loaders` only proves the other families still *register*. Before touching shared
engine code for CUDA, here is a real inference baseline with an existing Ming family:

```
family   ming3_tts (0.5B q8_0 package, untouched by this work)
command  audiocpp_cli --family ming3_tts --task tts --backend cpu \
           --model models/ming3-tts/ming-3.0-tts-0.5b-q8_0.gguf \
           --text "The garden will remain open until sunset." --out smoke/ming3_tts-baseline.wav

result   110250 samples, 44100 Hz, 2.50 s, rms 0.13250, peak 1.0000
time     4.6 s wall (includes model load)
```

That is the comparison point: after any shared-engine GPU change, rerun this exact
command and expect the same 2.50 s of audio at a comparable level (numeric differences
from a different backend are acceptable, silence or a crash is not).

Also worth running after such changes (cheap, different code paths):

- `ming3_audio` if a package is available locally — it exercises the omni backbone that
  shares more infrastructure than `ming3_tts`;
- the `ming3_tts_tiny` seeded patch, which is the primary GPU comparison.

## GPU plan and the comparison discipline

1. Add a KV cache to the tiny thinker (prefill once, then one token per patch) — the
   current full recompute is why a patch costs ~25 s.
2. Place the thinker, flow and VAE matmuls on CUDA.
3. **Gate on the seeded patch**: it must still reproduce the reference, and the
   established **4.8e-06** is a *baseline to measure against*, not a pass/fail threshold —
   CUDA arithmetic will differ somewhat (different reduction orders, TF32 questions).
   Report the new delta, the wall time per patch, and VRAM.
4. Re-run the `ming3_tts` smoke command above to confirm the shared engine is unaffected.

Only after the delta is understood should any threshold be set, and it should be stated
as "CPU vs CUDA differ by X", not "CUDA must hit the CPU number".
