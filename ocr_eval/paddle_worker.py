#!/usr/bin/env python
"""Runs in the PaddlePaddle env (paddlepaddle-gpu + paddleocr[doc-parser]); started by ocr_eval/paddle_vl.py.

Feeds pages to PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="vllm-server") and writes
res.markdown["markdown_texts"]. manifest = [[image_path, out_md_path], ...]
"""
import argparse
import json
import os
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--server-url", required=True, help="vLLM OpenAI base URL, e.g. http://127.0.0.1:8123/v1")
    ap.add_argument("--device", default="gpu:0", help="device for the layout model")
    args = ap.parse_args()

    from paddleocr import PaddleOCRVL

    jobs = json.load(open(args.manifest, encoding="utf-8"))
    pipeline = PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="vllm-server",
                           vl_rec_server_url=args.server_url, device=args.device)
    t0, n_ok, n_err = time.time(), 0, 0
    for i, (img_path, out_path) in enumerate(jobs, 1):
        try:
            parts = []
            for res in pipeline.predict(img_path):
                # pretty=False: plain Markdown, without PaddleX's centring <div>s and table styling (text,
                # HTML tables and LaTeX are unchanged); the text-matching scorers should not see decorations.
                md = res._to_markdown(pretty=False) if hasattr(res, "_to_markdown") else res.markdown
                parts.append(md["markdown_texts"] if isinstance(md, dict) else str(md))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(parts))
            n_ok += 1
        except Exception:
            n_err += 1
            print(f"[paddle-worker] FAILED {img_path}:\n{traceback.format_exc()}", flush=True)
        if i % 20 == 0 or i == len(jobs):
            el = time.time() - t0
            print(f"[paddle-worker] {i}/{len(jobs)} done ({n_err} failed), {el / 60:.1f} min", flush=True)
    sys.exit(1 if n_ok == 0 else 0)


if __name__ == "__main__":
    main()
