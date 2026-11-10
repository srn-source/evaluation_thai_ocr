#!/usr/bin/env python
"""Inference for one model on one benchmark. run_all_benchmarks.py calls this in the inference env; you can
also run it directly:

    python -m ocr_eval.infer --bench mdp --model-key typhoon-ocr2-2b \
        --data benchmarks/MultimodalOCR/MDPBench/MDPBench_dataset/MDPBench_img_public --out preds/mdp

Only the input pages and output names differ per benchmark; the model is always run from its registry
entry (ocr_eval/models.py):
  mdp     MDPBench        images whose names start with en_ / th_  -> <out>/<stem>.md
  olmocr  olmOCR-bench    page 1 of each PDF, rendered at 2048 px  -> <out>/<subdir>/<name>_pg1_repeat1.md
  thai    ThaiOCRBench    scb10x/ThaiOCRBench test split; the prompt is each item's question -> <out> (JSON)
  sea     SEA-DocBench    images whose ground-truth language is en / th -> <out>/<stem>.md
  omni    OmniDocBench    all images -> <out>/<stem>.md
"""
import argparse
import base64
import glob
import io
import json
import os
import sys

from PIL import Image
from tqdm import tqdm

from . import engine
from .models import MODELS, resolve

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def list_images(folder):
    paths = sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True))
    paths = [p for p in paths if p.lower().endswith(IMAGE_EXTS)]
    if not paths:
        raise SystemExit(f"No images found under {folder}")
    return paths


def image_jobs(paths, out_dir, force):
    jobs = []
    for p in paths:
        out = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".md")
        if force or not os.path.exists(out):
            jobs.append((out, p))
    return jobs


def jobs_mdp(args):
    prefixes = tuple(f"{lang}_" for lang in args.langs)        # MDPBench file names start with the language
    paths = [p for p in list_images(args.data) if os.path.basename(p).startswith(prefixes)]
    print(f"[mdp] languages {args.langs}: {len(paths)} images")
    return image_jobs(paths, args.out, args.force), None


def jobs_docbench(args):
    paths = list_images(args.data)
    if args.gt_json:                                           # SEA-DocBench: keep the GT languages we score
        allowed = set()
        for page in json.load(open(args.gt_json, encoding="utf-8")):
            if page["page_info"]["page_attribute"].get("language") in args.langs:
                allowed.add(page["page_info"]["image_path"])
        paths = [p for p in paths if os.path.basename(p) in allowed]
        print(f"[{args.bench}] languages {args.langs}: {len(paths)} images")
    return image_jobs(paths, args.out, args.force), None


def jobs_olmocr(args):
    """olmOCR-bench ships one-page PDFs; render page 1 with olmocr's own renderer (pdftoppm), longest side
    2048 px, then run the model on that image."""
    sys.path.insert(0, args.olmocr_repo)
    from olmocr.data.renderpdf import render_pdf_to_base64png

    def open_pdf(pdf):
        b64 = render_pdf_to_base64png(pdf, page_num=1, target_longest_image_dim=2048)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    pdf_root = os.path.join(args.data, "pdfs")
    pdfs = sorted(glob.glob(os.path.join(pdf_root, "**", "*.pdf"), recursive=True))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {pdf_root}")
    jobs = []
    for pdf in pdfs:
        rel = os.path.relpath(os.path.dirname(pdf), pdf_root)
        out = os.path.join(args.out, rel, os.path.splitext(os.path.basename(pdf))[0] + "_pg1_repeat1.md")
        if args.force or not os.path.exists(out):
            jobs.append((out, pdf))
    print(f"[olmocr] {len(pdfs)} PDFs")
    return jobs, open_pdf


def vqa_clean(text: str) -> str:
    """ThaiOCRBench answers are free text: strip whitespace and a stray <think> block, nothing else."""
    t = (text or "").strip()
    if t.startswith("<think>"):
        end = t.find("</think>")
        t = t[end + len("</think>"):].strip() if end != -1 else t
    return t


def run_thai(spec, args):
    """ThaiOCRBench: each item's own question is the prompt (wrapped as in the benchmark's run_inference.py),
    and the answer is only stripped, not converted. Model loading, image preprocessing and decoding are the
    same as for page parsing."""
    if spec.get("runner"):
        raise SystemExit(f"{args.model_key} has no free-form question mode; ThaiOCRBench does not apply")
    from datasets import load_dataset

    ds = load_dataset("scb10x/ThaiOCRBench")["test"]
    ds = ds.map(lambda x: {
        "question": "Instruction: " + x["question"].strip() + "\nตอบเฉพาะข้อความตามภาพตรง ๆ อย่างกระชับ โดยไม่ใส่คำอธิบายเพิ่มเติม"
        if isinstance(x["question"], str) else x["question"],
        "answer": x["answer"].strip() if isinstance(x["answer"], str) else x["answer"],
    })
    llm = engine.build_engine(spec, args)
    sampling = engine.build_sampling(spec)
    extra = engine.chat_kwargs(spec)
    preds, n, bs = [], len(ds), 32
    for start in tqdm(range(0, n, bs), desc=f"{args.model_key} ThaiOCRBench"):
        batch, refs = [], []
        for item in ds.select(range(start, min(start + bs, n))):
            try:
                img = spec["preprocess"](item["image"].convert("RGB"))
            except Exception as e:
                print(f"Skipping id={item.get('Id')} ({type(e).__name__}: {e})")
                continue
            batch.append(engine.build_messages(spec, img, item["question"], spec.get("vqa_system_prompt")))
            refs.append(item)
        if not batch:
            continue
        for res, item in zip(llm.chat(batch, sampling, use_tqdm=False, **extra), refs):
            preds.append({"type": item["Task"], "id": item["Id"], "question": item["question"],
                          "answers": [item["answer"]], "predict": vqa_clean(res.outputs[0].text)})
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:               # written after every batch
            json.dump(preds, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, choices=["mdp", "olmocr", "thai", "sea", "omni"])
    ap.add_argument("--model-key", required=True, choices=list(MODELS), help="registry entry (ocr_eval/models.py)")
    ap.add_argument("--model", default=None, help="weights: a Hugging Face id or a local dir (default: the entry's repo)")
    ap.add_argument("--data", default=None, help="mdp/sea/omni: image folder; olmocr: bench_data folder (thai: unused)")
    ap.add_argument("--out", required=True, help="output folder (thai: output JSON file)")
    ap.add_argument("--langs", nargs="+", default=["en", "th"], help="mdp/sea: languages to run (default en th)")
    ap.add_argument("--gt-json", default=None, help="sea: ground_truth.json, to select pages by language")
    ap.add_argument("--olmocr-repo", default=None, help="olmocr: clone of allenai/olmocr (for its PDF renderer)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None, help="vLLM override, e.g. 0.75 on a shared GPU")
    ap.add_argument("--enforce-eager", action="store_true", help="vLLM eager mode (no torch.compile / CUDA graphs)")
    ap.add_argument("--upscale", type=int, default=0, metavar="PX",
                    help="enlarge images whose long side is under PX px to PX before the model's own preprocessing")
    ap.add_argument("--max-pixels", type=int, default=None, help="override the processor's max_pixels")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=200, help="pages per llm.chat call")
    ap.add_argument("--save-raw", action="store_true", help="also write each raw model output as <name>.raw.txt")
    ap.add_argument("--force", action="store_true", help="regenerate outputs that already exist")
    args = ap.parse_args()

    spec = resolve(args.model_key, upscale=args.upscale, max_pixels=args.max_pixels)
    args.model = args.model or spec["repo"]
    print(f"[{args.bench}] model={args.model_key} weights={args.model}", flush=True)

    if args.bench == "thai":
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
        return run_thai(spec, args)
    jobs, opener = {"mdp": jobs_mdp, "olmocr": jobs_olmocr, "sea": jobs_docbench, "omni": jobs_docbench}[args.bench](args)
    if not jobs:
        print("Nothing to do: all outputs exist (use --force to regenerate).")
        return
    engine.run_pages(jobs, spec, args, open_image=opener)
    print(f"Wrote {len(jobs)} files under {args.out}")


if __name__ == "__main__":
    main()
