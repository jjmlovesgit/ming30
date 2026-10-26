"""Pre-flight checks for the no-Docker stack.

    python scripts/preflight.py [--config config/dev.server.json]

Verifies, without starting anything: the native engine binary, the CUDA toolkit
binaries its DLLs need, every GGUF referenced by the engine config, the Python
dependencies, ffmpeg, and whether the ports are free. Exit code 0 means ready.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name:<28} {detail}")


def free_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dev.server.json")
    parser.add_argument("--runtime", default=str(ROOT.parent / "audio.cpp-0.9.3-cuda12.4"))
    parser.add_argument("--cuda-bin", default=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")
    parser.add_argument("--backend-port", type=int, default=8090)
    parser.add_argument("--port", type=int, default=8070)
    args = parser.parse_args()

    # engine binary and its CUDA DLLs
    engine = Path(args.runtime) / "audiocpp_server.exe"
    check("engine binary", engine.exists(), str(engine))
    cuda = Path(args.cuda_bin)
    check("CUDA bin", cuda.exists(), str(cuda))
    for dll in ("cublas64_12.dll", "cufft64_11.dll", "cudart64_12.dll"):
        present = (cuda / dll).exists() or (Path(r"C:\Windows\System32") / dll).exists()
        check(f"  {dll}", present, "found" if present else "missing — the engine will exit 0xC0000135")

    # nvidia-smi / GPU
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
            check("GPU", True, out.splitlines()[0] if out else "")
        except Exception as exc:  # noqa: BLE001
            check("GPU", False, str(exc))
    else:
        check("GPU", False, "nvidia-smi not on PATH")

    # model files referenced by the engine config
    config_path = ROOT / args.config
    if not config_path.exists():
        check("engine config", False, str(config_path))
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        check("engine config", True, f"{len(config.get('models', []))} model entries")
        for entry in config.get("models", []):
            path = Path(entry["path"])
            check(f"  {entry['id']}", path.exists(), f"{path} ({path.stat().st_size // (1024**2)} MB)" if path.exists() else str(path))

    # python deps for the adapter
    for module in ("fastapi", "uvicorn", "httpx", "multipart", "pydantic"):
        check(f"python {module}", importlib.util.find_spec(module) is not None)
    check("python version", sys.version_info >= (3, 10), sys.version.split()[0])

    # ffmpeg for mp3/opus output
    ffmpeg = shutil.which("ffmpeg")
    check("ffmpeg", ffmpeg is not None, ffmpeg or "mp3/opus/flac will be unavailable")

    # ports
    check(f"port {args.backend_port} free", free_port(args.backend_port), "engine")
    check(f"port {args.port} free", free_port(args.port), "adapter")

    failed = [result for result in RESULTS if not result[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("not ready:", ", ".join(name.strip() for name, _, _ in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
