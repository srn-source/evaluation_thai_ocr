"""Run a model spec (ocr_eval/models.py) with offline vLLM: engine, decoding, request format, output files."""
import base64
import io
import os

from PIL import Image
from tqdm import tqdm


def pil_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def write_output(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_engine(spec, args):
    from vllm import LLM

    kwargs = dict(model=args.model, max_model_len=spec["max_model_len"], tensor_parallel_size=args.tensor_parallel_size,
                  limit_mm_per_prompt={"image": 1})
    if spec.get("mm_processor_kwargs") is not None:
        kwargs["mm_processor_kwargs"] = spec["mm_processor_kwargs"]
    if spec.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    kwargs.update(spec.get("llm_extra", {}))
    if args.gpu_memory_utilization:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    print(f"Loading {args.model} with vLLM: {kwargs}", flush=True)
    return LLM(**kwargs)


def build_sampling(spec):
    from vllm import SamplingParams

    kwargs = dict(temperature=spec["temperature"], top_p=spec["top_p"], max_tokens=spec["max_tokens"])
    if spec.get("stop"):
        kwargs["stop"] = spec["stop"]
    if spec.get("seed") is not None:
        kwargs["seed"] = spec["seed"]
    kwargs.update(spec.get("sampling_extra", {}))
    return SamplingParams(**kwargs)


def chat_kwargs(spec):
    """Extra llm.chat() arguments: thinking switch and message content format."""
    extra = {}
    if spec.get("chat_template_kwargs"):
        extra["chat_template_kwargs"] = spec["chat_template_kwargs"]
    if spec.get("chat_template_content_format"):
        extra["chat_template_content_format"] = spec["chat_template_content_format"]
    return extra


def build_messages(spec, image: Image.Image, text: str, system_prompt: str = None):
    """One OpenAI-format conversation: optional system prompt, then the image and the text."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": pil_to_data_uri(image)}},
        {"type": "text", "text": spec.get("text_prefix", "") + text},
    ]})
    return msgs


def build_request(spec, image: Image.Image, text: str, system_prompt: str = None):
    """A chat conversation, or for raw_prompt models a plain prompt with the image (no chat template)."""
    if spec.get("raw_prompt"):
        return {"prompt": spec.get("text_prefix", "") + text, "multi_modal_data": {"image": image}}
    return build_messages(spec, image, text, system_prompt)


def generate(llm, spec, batch, sampling, extra, use_tqdm=True):
    if spec.get("raw_prompt"):
        return llm.generate(batch, sampling, use_tqdm=use_tqdm, tokenization_kwargs={"add_special_tokens": False})
    return llm.chat(batch, sampling, use_tqdm=use_tqdm, **extra)


def postprocess(spec, raw: str, origin: Image.Image, image: Image.Image) -> str:
    if spec.get("postprocess_ctx"):
        return spec["postprocess_ctx"](raw, origin, image)
    return spec["postprocess"](raw)


def run_pages(jobs, spec, args, open_image=None):
    """Parse pages: jobs = [(out_path, source)]; `open_image(source)` returns the page as PIL
    (default: open an image file). Writes one Markdown file per page; returns the sources it could not read."""
    open_image = open_image or (lambda src: Image.open(src).convert("RGB"))
    if spec.get("runner") == "paddleocr_vl":
        from .paddle_vl import run_paddleocr_vl
        run_paddleocr_vl(jobs, spec, args, open_image)
        return []

    llm = build_engine(spec, args)
    sampling = build_sampling(spec)
    extra = chat_kwargs(spec)
    skipped = []
    # chunks of 200 pages: bounded memory, files written as we go
    for i in tqdm(range(0, len(jobs), args.chunk_size), desc=f"{args.model_key}", unit="chunk"):
        batch, outs, geoms = [], [], []
        for out_path, src in jobs[i:i + args.chunk_size]:
            try:
                origin = open_image(src)
                img = spec["preprocess"](origin)
            except Exception as e:
                print(f"Skipping unreadable page {src} ({type(e).__name__}: {e})")
                skipped.append(src)
                continue
            batch.append(build_request(spec, img, spec["prompt"], spec.get("system_prompt")))
            outs.append(out_path)
            geoms.append((origin, img))
        if not batch:
            continue
        results = generate(llm, spec, batch, sampling, extra)
        for out_path, res, (origin, img) in zip(outs, results, geoms):
            raw = res.outputs[0].text
            write_output(out_path, postprocess(spec, raw, origin, img))
            if args.save_raw:
                write_output(os.path.splitext(out_path)[0] + ".raw.txt", raw)
    return skipped
