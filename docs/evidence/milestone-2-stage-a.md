# Milestone 2, stage A — prompt token ids match the Python reference

The cheapest gate, and the one that isolates tokenizer/template mistakes from model
maths. Run against the full BF16 package.

```
[ming3_tts_tiny] structural package report
  tensors         : 2141
  prompt stage A  : text=The garden will remain open until sunset.
    user_prefix      ids=19
    speaker_block    ids=3
    text_prefix      ids=5
    target_text      ids=8
    assistant_prefix ids=5
    audio_start      ids=1
    total              =41 (segments 41)
    speaker_position   =20
    ids=157151,39,116171,157152,7775,8941,12050,2901,366,268,2538,7787,13,198,220,
        18233,62,16,25,157175,157164,157176,198,9333,3967,25,198,678,10635,627,4206,
        2305,3334,39563,13,157151,8469,7342,5468,157152,157165
```

Comparison against `out/reference/stage-a-segments.json`:

```
reference ids : 41
cpp ids       : 41
PASS: stage A prompt ids match the reference exactly
sha256(cpp)   : ac28cb883ed49d19cb0061ff590ddb267bc875953918d707d55414cc15366034
sha256(ref)   : ac28cb883ed49d19cb0061ff590ddb267bc875953918d707d55414cc15366034
```

## What this establishes

- The `BailingMoe` pre-tokenizer reproduces the checkpoint tokenizer for this input.
- The legacy prompt template is byte-exact: `<role>HUMAN</role>…\n  speaker_1:` +
  `<spk><audioPatch></spk>` + `\n Text input:\n` + text + `<role>ASSISTANT</role>` +
  `<audio>`.
- Special token ids are wired correctly: `<spk>` 157175, `<audioPatch>` 157164,
  `</spk>` 157176, `<audio>` 157165.
- Segment-wise encoding matches the reference's segment-wise encoding — tokenising a
  joined string would be a different thing, and this test would catch it.
- `speaker_embedding_position = 20` agrees.

## What it does not establish

Nothing about the thinker, flow or VAE. This is input plumbing only.

Stages and where they can be compared:

| stage | reference source | status |
|---|---|---|
| A prompt ids | checkpoint `prompt_contract.py` (pure Python) | **PASS, exact** |
| B thinker output | needs `atorch` + `fla` → Linux/WSL only | reference not yet captureable here |
| C flow latents | `CFM.py`/`dit.py`/`aggregator.py` run on Windows | needs the same captured inputs on both sides |
| D VAE audio | `modeling_audio_vae.py` runs on Windows | same requirement |

Running C and D standalone does not substitute for a thinker reference: they must be
fed identical captured tensors on both sides to mean anything, and the thinker is the
stage that actually needs a Linux reference to compare against.
