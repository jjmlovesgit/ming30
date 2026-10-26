# Provenance

Reference copies for the Ming-3.0-tts-tiny port. All files upstream, MIT licensed,
retrieved 2026-10-26. They are here so the C++ port has a byte-exact target for
prompt assembly, task conditioning and decode alignment.

| file | source |
|---|---|
| `prompt_contract.py` | https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny (repo root) |
| `config.json` | same repo, repo root |
| `model.safetensors.index.json` | same repo, repo root (tensor map only) |
| `task_protocol.py` | https://github.com/inclusionAI/Ming-3.0-tts/blob/main/ming_tiny/task_protocol.py |
| `inference.py` | https://github.com/inclusionAI/Ming-3.0-tts/blob/main/ming_tiny/inference.py |
| `api.py` | https://github.com/inclusionAI/Ming-3.0-tts/blob/main/ming_tiny/api.py |

Not included: model weights (`model-*-of-00030.safetensors`, ~16 GB). They stay
outside version control.
