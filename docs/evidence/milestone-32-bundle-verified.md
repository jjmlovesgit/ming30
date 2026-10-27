# Portable copy of the audio.cpp branch: verified and hashed

Decision: keep the local branch and the bundle, no fork, no push. A public fork is an external
publication, and the port has not passed its final feature and GPU gates. The upstream remote
(`0xShug0/audio.cpp`) is untouched.

## Artifact

```
file      : out/ming3-tts-tiny.bundle
size      : 130.3 MB
SHA-256   : D49E89C0A705EE1BC1B0D8F4E63A2E0320C97189B9A9F773B2E8C782A90C4FDA
ref       : refs/heads/ming3-tts-tiny -> b7f4fa91c872ba6515f73b917b4a42fb8b692731
```

## Verification performed

```
git bundle verify  -> "is okay", "The bundle records a complete history"
git clone --branch ming3-tts-tiny <bundle> <scratch dir>
  clone exit        : 0
  restored branch   : ming3-tts-tiny
  restored HEAD     : b7f4fa91 ming3_tts_tiny: VAE encoder (WAV reference latents)
  family sources    : assets, encoder, flow, gpu_bench.cu, gpu_moe.cu, gpu_residency.cu,
                      loop, prompt, session, thinker, vae  (11 files)
  encoder.cpp       : present
```

So the branch restores with complete history, including the encoder work that was untracked
before the bundle was made.

## Caveat, stated plainly

A bundle on the same disk is **not** protection against disk failure. It is a portable copy — it
can be moved to another machine or medium, and it restores without GitHub. Real redundancy needs
a second physical location (external drive, another machine, or an encrypted archive off-disk),
which is a decision for the user, not something the agent should do silently.

## Upstream

`origin` remains `https://github.com/0xShug0/audio.cpp.git` and has not been contacted: no fetch,
no push, no fork. When the gates pass and publication is decided, the options remain a fork under
the user's account or a patch series offered to upstream.
