"""PaddleOCR-VL-1.6 (arXiv:2606.03264) with its official page-parsing pipeline.

The VLM only reads layout crops with fixed task prompts ("OCR:", "Table Recognition:", ...). Page parsing is
PP-DocLayoutV3 layout detection -> per-crop VLM -> PaddleOCR post-processing -> Markdown, and the docs say
that sending whole pages to the VLM "is not equivalent" to the reported numbers. So we:
  1. start `vllm serve PaddlePaddle/PaddleOCR-VL-1.6` (flags from the official vLLM recipe) on the run's GPU,
  2. run paddle_worker.py in a PaddlePaddle env (PADDLE_PYTHON) with
     PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="vllm-server") over the pages,
  3. stop the server.
Decoding is the pipeline's own; nothing is overridden.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

PADDLE_PYTHON = os.environ.get("PADDLE_PYTHON", "")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paddle_worker.py")
SERVED_NAME = "PaddleOCR-VL-1.6-0.9B"   # the pipeline's default vl_rec_model_name
SERVE_FLAGS = ["--trust-remote-code", "--max-num-batched-tokens", "16384", "--no-enable-prefix-caching",
               "--mm-processor-cache-gb", "0"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _serve_flags() -> list:
    """The pipeline sends per-request mm_processor_kwargs; newer vLLM (e.g. 0.33) rejects them unless the
    server is started with --trust-request-mm-kwargs, a flag older versions (0.27, 0.30) do not have."""
    flags = list(SERVE_FLAGS)
    try:
        import inspect
        from vllm.entrypoints.common import cli_args
        if "trust_request_mm_kwargs" in inspect.getsource(cli_args):
            flags.append("--trust-request-mm-kwargs")   # the only client is our own worker on 127.0.0.1
    except Exception:
        pass
    return flags


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthy(url, proc, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"vllm serve did not become healthy within {timeout_s}s ({url})")


def run_paddleocr_vl(jobs, spec, args, open_image):
    if not PADDLE_PYTHON or not os.path.exists(PADDLE_PYTHON):
        raise SystemExit("paddleocr-vl-1.6 needs a PaddlePaddle env; set PADDLE_PYTHON to its python "
                         "(see README: conda create -n paddle python=3.11 ...)")
    tmp = tempfile.mkdtemp(prefix="paddleocr_vl_")
    manifest = []
    for k, (out_path, src) in enumerate(jobs):
        if isinstance(src, str) and src.lower().endswith(IMAGE_EXTS):
            img_path = src
        else:  # e.g. an olmOCR-bench PDF: render it and hand the pipeline a PNG
            img_path = os.path.join(tmp, f"page_{k:06d}.png")
            open_image(src).save(img_path)
        manifest.append([img_path, out_path])
    manifest_path = os.path.join(tmp, "jobs.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    port = _free_port()
    gpu_util = args.gpu_memory_utilization or spec["llm_extra"]["gpu_memory_utilization"]
    serve = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model, "--host", "127.0.0.1",
             "--port", str(port), "--served-model-name", SERVED_NAME, "model",
             "--gpu-memory-utilization", str(gpu_util), "--max-model-len", str(spec["max_model_len"]),
             "--tensor-parallel-size", str(args.tensor_parallel_size), *_serve_flags()]
    print(f"[paddleocr-vl] starting the VLM server: {' '.join(serve)}", flush=True)
    log = open(os.path.join(tmp, "vllm_serve.log"), "w")
    server = subprocess.Popen(serve, stdout=log, stderr=subprocess.STDOUT)
    try:
        _wait_healthy(f"http://127.0.0.1:{port}/health", server)
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        env["PATH"] = f"{os.path.dirname(PADDLE_PYTHON)}:{env.get('PATH', '')}"
        rc = subprocess.call([PADDLE_PYTHON, WORKER, "--manifest", manifest_path,
                              "--server-url", f"http://127.0.0.1:{port}/v1", "--device", "gpu:0"], env=env)
        if rc != 0:
            raise RuntimeError(f"PaddleOCR pipeline worker failed (exit {rc}); vLLM log: {log.name}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except Exception:
            server.kill()
        log.close()
    missing = [o for o, _ in jobs if not os.path.exists(o)]
    if missing:
        print(f"[paddleocr-vl] WARNING: {len(missing)} pages produced no output (see the worker log)", flush=True)
