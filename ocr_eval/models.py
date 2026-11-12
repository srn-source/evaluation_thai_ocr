"""The models we compare, each run the way its authors run it.

Every entry fixes the prompt, image preprocessing, decoding settings and the conversion of the model
output to Markdown. These come from each model's card or official inference code. All five benchmarks use
the same entry, so a model is run identically everywhere; only the input pages and the output file names
differ per benchmark (see ocr_eval/infer.py).

Model                   key                      notes
Typhoon OCR V2 2B       typhoon-ocr2-2b          layout JSON prompt -> Markdown (header/footer dropped)
Typhoon OCR V2 0.8B     typhoon-ocr2-0.8b        same recipe as the 2B
Infinity-Parser2-Flash  infinity-parser2-flash   same prompt and conversion as Typhoon OCR V2
Typhoon OCR 1.5         typhoon-ocr1.5-2b        Markdown prompt, image long side <= 1800 px
Chandra OCR 2           chandra-ocr-2            needs `pip install chandra-ocr` (prompt, resize, HTML -> Markdown)
dots.mocr               dots.mocr                needs a clone of github.com/rednote-hilab/dots.mocr (DOTS_MOCR_REPO)
PaddleOCR-VL-1.6        paddleocr-vl-1.6         official 2-stage pipeline in a separate PaddlePaddle env
OvisOCR2                ovisocr2                 Markdown prompt from its model card
OpenThai2.0-OCR-Markdown openthai2.0-ocr-markdown Nemotron 3.5 Parse fine-tune; vLLM >= 0.33; Parse's own postprocessing
Nemotron 3.5 Parse      nemotron-3.5-parse       the base model of OpenThai2.0; same recipe
DeepSeek-OCR-2          deepseek-ocr-2           grounding Markdown prompt, n-gram repeat blocker, official eval post-processing
"""
import importlib.util
import json
import os
import re
import sys
import types

from PIL import Image

# --------------------------------------------------------------------------------------------------------
# Typhoon OCR V2 and Infinity-Parser2: the doc2json prompt (verbatim from the Infinity-Parser2 README).
# The model returns a JSON list of layout elements; their text fields are joined into Markdown.
# --------------------------------------------------------------------------------------------------------
DOC2JSON_PROMPT = """- Extract layout information from the provided PDF image.
- For each layout element, output its bbox, category, and the text content within the bbox.
- Bbox format: [x1, y1, x2, y2].
- Allowed layout categories: ['header', 'title', 'text', 'figure', 'table', 'formula', 'figure_caption', 'table_caption', 'formula_caption', 'figure_footnote', 'table_footnote', 'page_footnote', 'footer'].
- Text extraction and formatting:
  1) For 'figure', the text field must be an empty string.
  2) For 'formula', format text as LaTeX.
  3) For 'table', format text as HTML.
  4) For all other categories (e.g., text, title), format text as Markdown.
- The output text must be exactly the original text from the image, with no translation or rewriting.
- Sort all layout elements in human reading order.
- Final output must be a single JSON object."""

HEADER_FOOTER = ("header", "footer", "page_footnote")


def _extract_json_content(text: str) -> str:
    """Port of infinity_parser2.utils.extract_json_content: the JSON inside a ```json fence
    (or after an unclosed fence)."""
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    partial = re.search(r"```json\n(.*)", text, re.DOTALL)
    if partial:
        return partial.group(1).strip()
    return text


def _truncate_last_incomplete_element(text: str) -> str:
    """Port of infinity_parser2.utils.truncate_last_incomplete_element: drop the last, cut-off element
    of an output that stopped at the token limit, so the rest parses."""
    needs_truncation = len(text) > 65536 or not text.rstrip().endswith("]")
    if not needs_truncation:
        return text
    if text.count('{"bbox":') <= 1:
        return text
    truncated = text[:text.rfind('{"bbox":')].rstrip()
    if truncated.endswith(","):
        truncated = truncated[:-1] + "]"
    return truncated


# One malformed number (e.g. a bbox value "630.6.62") makes json.loads fail for the whole page, and scoring
# the raw JSON then costs nearly the whole page. Salvage instead: elements are always written as
# {"bbox": ..., "category": "...", "text": "..."}, so pull (category, text) pairs out of the broken string.
_SALVAGE_ELEMENT_RE = re.compile(r'"category"\s*:\s*"([^"]*)"\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
# An output cut at max_tokens leaves the last "text" string open; it often still holds real text.
_SALVAGE_TAIL_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)$', re.DOTALL)


def _strip_repeated_tail(text: str) -> str:
    """Remove a tail that repeats the same snippet 10 or more times (a repetition loop)."""
    n = len(text)
    if n < 200:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count, i = 0, n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[: n - length * (count - 1)]
    return text


def _salvage_lines(text: str, keep_header_footer: bool) -> list:
    lines = []
    for category, escaped in _SALVAGE_ELEMENT_RE.findall(text):
        if not keep_header_footer and category in HEADER_FOOTER:
            continue
        try:
            content = json.loads(f'"{escaped}"')
        except Exception:
            content = escaped
        if content:
            lines.append(content)
    tail = _SALVAGE_TAIL_RE.search(text)
    if tail:
        try:
            content = json.loads(f'"{tail.group(1)}"')
        except Exception:
            content = tail.group(1).replace("\\n", "\n").replace('\\"', '"')
        content = _strip_repeated_tail(content)
        if content:
            lines.append(content)
    if not lines and not keep_header_footer:
        # every element was a header or footer (early stop): keep those few lines instead of raw JSON
        return _salvage_lines(text, keep_header_footer=True)
    return lines


def doc2json_to_markdown(raw: str) -> str:
    """Infinity-Parser2's doc2json -> Markdown conversion (postprocess_doc2json_result +
    convert_json_to_markdown with keep_header_footer=False): header, footer and page_footnote elements are
    left out. Box coordinates are not used, because they do not change the Markdown."""
    text = _truncate_last_incomplete_element(_extract_json_content(raw))
    try:
        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError("not a JSON list")
        lines = []
        for sub in items:
            if "text" not in sub or not sub["text"]:
                continue
            if sub.get("category") not in HEADER_FOOTER:
                lines.append(sub["text"])
        if not lines:  # every element is a header or footer: keep them rather than writing raw JSON
            lines = [sub["text"] for sub in items if isinstance(sub, dict) and sub.get("text")]
        return "\n\n".join(lines) if lines else text
    except Exception:
        lines = _salvage_lines(text, keep_header_footer=False)
        return "\n\n".join(lines) if lines else text


# --------------------------------------------------------------------------------------------------------
# Typhoon OCR 1.5 (prompt from its model card; the model was trained on pages up to 1800 px)
# --------------------------------------------------------------------------------------------------------
TYPHOON15_PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in English.
</figure>

- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""


def resize_longest(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    s = max_dim / max(w, h)
    return img.resize((round(w * s), round(h * s)), Image.LANCZOS)


# --------------------------------------------------------------------------------------------------------
# Chandra OCR 2: prompt, resize and HTML -> Markdown come from the `chandra-ocr` package itself.
# --------------------------------------------------------------------------------------------------------
def _chandra():
    try:
        from chandra.model.util import scale_to_fit
        from chandra.output import parse_markdown
        from chandra.prompts import PROMPT_MAPPING
    except ImportError:
        raise SystemExit("chandra-ocr-2 needs the chandra package in the inference env: pip install chandra-ocr")
    return scale_to_fit, parse_markdown, PROMPT_MAPPING


def chandra_preprocess(im: Image.Image) -> Image.Image:
    return _chandra()[0](im)


def chandra_postprocess(raw: str) -> str:
    try:
        return _chandra()[1](raw)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Warning: parse_markdown failed ({type(e).__name__}); writing raw.")
        return raw or ""


# --------------------------------------------------------------------------------------------------------
# dots.mocr (arXiv:2603.13032), as in the official dots_mocr/parser.py "parse" path behind their benchmark
# numbers:
#   prompt      prompt_layout_all_en (layout JSON)
#   preprocess  re-render at 200 dpi through fitz, then smart_resize to [3136, 11289600] px (factor 28)
#   request     the text part starts with the image placeholders "<|img|><|imgpad|><|endofimg|>" and is
#               sent with chat_template_content_format="string" (model/inference.py)
#   decoding    temperature 0.1, top_p 1.0, max 16384 new tokens
#   postprocess post_process_output -> layoutjson2md(no_page_hf=True)  (headers and footers dropped)
# All of this is imported from a clone of the dots.mocr repo, so nothing is re-implemented here.
# --------------------------------------------------------------------------------------------------------
DOTS_MIN_PIXELS, DOTS_MAX_PIXELS, DOTS_DPI = 3136, 11289600, 200
DOTS_IMG_PREFIX = "<|img|><|imgpad|><|endofimg|>"
DOTS_PROMPT_LAYOUT_ALL = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
_DOTS = {}


def _dots_utils():
    if "mod" not in _DOTS:
        repo = os.environ.get("DOTS_MOCR_REPO", os.path.join(os.path.dirname(__file__), "..", "benchmarks", "dots.mocr"))
        if not os.path.exists(os.path.join(repo, "dots_mocr", "utils", "prompts.py")):
            raise SystemExit(f"dots.mocr repo not found at {repo}: run setup_benchmarks.sh or set DOTS_MOCR_REPO")
        sys.path.insert(0, os.path.abspath(repo))
        sys.modules.setdefault("cairosvg", types.ModuleType("cairosvg"))  # only used by the image->svg mode
        from dots_mocr.utils import prompts, image_utils, layout_utils, format_transformer
        assert prompts.dict_promptmode_to_prompt["prompt_layout_all_en"] == DOTS_PROMPT_LAYOUT_ALL, \
            "dots.mocr changed prompt_layout_all_en upstream; check the pinned commit in setup_benchmarks.sh"
        _DOTS["mod"] = types.SimpleNamespace(image_utils=image_utils, layout_utils=layout_utils,
                                             format_transformer=format_transformer)
    return _DOTS["mod"]


def dots_preprocess(im: Image.Image) -> Image.Image:
    u = _dots_utils()
    im = u.image_utils.get_image_by_fitz_doc(im, target_dpi=DOTS_DPI)
    return u.image_utils.fetch_image(im, min_pixels=DOTS_MIN_PIXELS, max_pixels=DOTS_MAX_PIXELS)


_DOTS_B64_IMG = re.compile(r"!\[\]\(data:image/[a-z]+;base64,[A-Za-z0-9+/=]+\)")


def dots_postprocess(raw: str, origin: Image.Image, image: Image.Image) -> str:
    """post_process_output -> layoutjson2md(no_page_hf=True), as in parser.py. The one change: Picture
    cells, which upstream embeds as base64 PNG crops, are written as an empty `![]()` (scorers ignore
    images; this keeps the files small)."""
    u = _dots_utils()
    cells, filtered = u.layout_utils.post_process_output(raw, "prompt_layout_all_en", origin, image,
                                                         min_pixels=DOTS_MIN_PIXELS, max_pixels=DOTS_MAX_PIXELS)
    if filtered:  # the JSON could not be parsed even after cleaning: upstream writes the salvaged text
        return cells if isinstance(cells, str) else (raw or "")
    md = u.format_transformer.layoutjson2md(origin, cells, text_key="text", no_page_hf=True)
    return _DOTS_B64_IMG.sub("![]()", md)


# --------------------------------------------------------------------------------------------------------
# OvisOCR2 (ATH-MaaS/OvisOCR2), from the model card's OvisOCR2Parser: prompt (note the leading "\n"), greedy,
# 16384 new tokens, thinking off, processor pixels 448^2 .. 2880^2, gdn_prefill_backend="triton".
# Post-processing = the card's default parse(): drop <img src="images/bbox_..."> blocks, then trim a
# repeated tail on long outputs.
# --------------------------------------------------------------------------------------------------------
OVIS_OCR2_PROMPT = (
    '\nExtract all readable content from the image in natural human reading order and output the result '
    'as a single Markdown document. For charts or images, represent them using an HTML image tag: '
    '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are '
    'bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: '
    '<table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text '
    'without translation or paraphrasing.'
)


def _ovis_clean_truncated_repeats(text, min_text_len=8000, max_period=200, min_period=1,
                                  min_repeat_chars=100, min_repeat_times=5):
    """Port of OvisOCR2Parser._clean_truncated_repeats: keep one copy of a repeated tail."""
    n = len(text)
    if n < min_text_len:
        return text
    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len, idx = 1, n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1
        total_len = match_len + unit_len
        repeat_times, tail_len = total_len // unit_len, total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len:]
    return text


def ovis_ocr2_postprocess(raw: str) -> str:
    text = (raw or "").strip()
    text = "\n\n".join(b for b in text.split("\n\n") if not b.strip().startswith('<img src="images/bbox_'))
    return _ovis_clean_truncated_repeats(text)


# --------------------------------------------------------------------------------------------------------
# NVIDIA Nemotron 3.5 Parse (0.63B, RADIO encoder + mBART decoder) and its Thai fine-tune
# OpenThai2.0-OCR-Markdown (iapp). The two repos differ only in the weights: config, processor, tokenizer
# and the nemotron_parse/ package are byte-identical. Both cards say to use NVIDIA's DocumentParser,
# mode="single" (box-and-class prompt, one pass per page). So, from nemotron_parse/inference.py:
#   prompt      PROMPT, a string of task tokens with no free text; offline vLLM puts one "<pad>" image
#               placeholder directly before it and tokenizes with add_special_tokens=False (docs/serving.md)
#   preprocess  none: the model's processor fits the page into 1664 x 2048 (no upscaling)
#   decoding    GenerationSettings() for the full page: temperature 0.2, top_k 50, top_p 1.0,
#               repetition_penalty 1.05, seed 0, max 8744 tokens, special tokens kept (the box/class tags)
#   postprocess clean_raw_output -> postprocess_document(...)["markdown"]  (tables as HTML; all block
#               classes kept, page headers and footers included, as DocumentParser returns them), then two
#               content-neutral normalizations: unit rowspan/colspan attributes dropped, sara am composed
# The postprocessing package (nemotron_parse/) ships inside the model repo and is imported from there.
# Native vLLM support for this architecture (RadioMBartPrefixForCausalLM) needs vLLM 0.33 or newer.
# --------------------------------------------------------------------------------------------------------
OPENTHAI_REPO = "iapp/OpenThai2.0-OCR-Markdown"
NEMOTRON_PARSE_REPO = "nvidia/Nemotron-3.5-Parse"
NEMOTRON_PARSE_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
_NP = {}


def _nemotron_parse_setup(model: str):
    """Import nemotron_parse/ from the weights dir (a fine-tuned copy may carry its own), else from the
    model's Hugging Face repo, or NVIDIA's for a local dir without it (only that folder is downloaded). It is loaded by file location, so the
    repo's other top-level files and folders never land on sys.path."""
    root = model if os.path.isdir(os.path.join(model, "nemotron_parse")) else None
    if root is None:
        from huggingface_hub import snapshot_download
        root = snapshot_download(NEMOTRON_PARSE_REPO if os.path.isdir(model) else model, allow_patterns=["nemotron_parse/*"])
    pkg = os.path.join(root, "nemotron_parse")
    if not os.path.exists(os.path.join(pkg, "postprocessing.py")):
        raise SystemExit(f"nemotron_parse/ not found in {root}")
    spec = importlib.util.spec_from_file_location("nemotron_parse", os.path.join(pkg, "__init__.py"),
                                                  submodule_search_locations=[pkg])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nemotron_parse"] = mod
    spec.loader.exec_module(mod)
    from nemotron_parse import inference, postprocessing
    assert inference.PROMPT == NEMOTRON_PARSE_PROMPT, "nemotron_parse changed its prompt; check the model repo"
    s = inference.GenerationSettings()   # DocumentParser's full-page settings, checked against ours
    ours = _NEMOTRON_PARSE
    assert (s.temperature, s.top_p, s.max_tokens, s.seed) == (ours["temperature"], ours["top_p"],
                                                             ours["max_tokens"], ours["seed"]) and \
        (s.top_k, s.repetition_penalty) == (ours["sampling_extra"]["top_k"],
                                            ours["sampling_extra"]["repetition_penalty"]), \
        f"nemotron_parse changed its default GenerationSettings ({s}); update the registry entry"
    _NP["post"] = postprocessing
    print(f"[nemotron-parse] nemotron_parse postprocessing from {pkg}", flush=True)


_UNIT_SPAN = re.compile(r'\s+(?:col|row)span="1"')


def _drop_unit_spans(markdown: str) -> str:
    """nemotron_parse writes rowspan="1" colspan="1" on every cell of most tables. That is the HTML default, so
    dropping it changes nothing in the table, but the benchmarks' table edit distance (SEA-DocBench's table NED)
    compares the HTML as text and counts those attributes as errors. TEDS is unaffected either way."""
    return re.sub(r"<t[dh]\b[^>]*>", lambda m: _UNIT_SPAN.sub("", m.group(0)), markdown)


def _compose_sara_am(text: str) -> str:
    """The tokenizer normalizes to NFKC and has no U+0E33 (Thai sara am) in its vocabulary, so the model always
    writes it as U+0E4D U+0E32. Both render the same; this puts the standard character back, which is what
    the benchmarks' references use (their text metrics do not normalize it)."""
    return text.replace("\u0e4d\u0e32", "\u0e33")


def nemotron_parse_postprocess(raw: str) -> str:
    """DocumentParser.parse(mode="single") without the HTTP call: clean the raw output, then render it."""
    if "post" not in _NP:
        raise RuntimeError("nemotron_parse is not loaded (setup was not called)")
    post = _NP["post"]
    options = post.PostprocessOptions()
    raw = post.clean_raw_output(raw or "", deduplicate=options.deduplicate)
    try:
        return _compose_sara_am(_drop_unit_spans(post.postprocess_document(raw, options=options)["markdown"]))
    except Exception as e:   # keep the page's text rather than nothing
        print(f"Warning: nemotron_parse postprocess failed ({type(e).__name__}: {e}); writing tag-stripped raw.")
        return re.sub(r"<x_[^>]+>|<y_[^>]+>|<class_[^>]+>", "", raw).strip()


# --------------------------------------------------------------------------------------------------------
# DeepSeek-OCR-2 (deepseek-ai/DeepSeek-OCR-2, arXiv:2601.20552), as in the official OmniDocBench script
# DeepSeek-OCR2-vllm/run_dpsk_ocr2_eval_batch.py + config.py (github.com/deepseek-ai/DeepSeek-OCR-2):
#   prompt      "<image>\n<|grounding|>Convert the document to markdown." as a plain prompt (no chat template)
#   preprocess  none here: vLLM's native processor does the official tiling (global view padded to 1024 px,
#               2-6 tiles of 768 px when a side is over 768 px; 257 + 144 x tiles image tokens)
#   decoding    greedy, max 8192 tokens (max_model_len 8192), special tokens kept, and the no-repeat n-gram
#               logits processor with ngram_size 40, window 90, <td>/</td> (128821, 128822) whitelisted
#   postprocess clean_formula, then every <|ref|>..<|/ref|><|det|>..<|/det|> grounding span removed
#               (image regions too) and runs of blank lines collapsed, exactly as the eval script writes .md
# Native vLLM support (DeepseekOCR2ForCausalLM) replaces the repo's custom vLLM 0.8.5 code.
# --------------------------------------------------------------------------------------------------------
DEEPSEEK_OCR2_PROMPT = "<|grounding|>Convert the document to markdown."
_DS_REF_DET = re.compile(r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.DOTALL)
_DS_EOS = "<｜end▁of▁sentence｜>"


def _ds_clean_formula(text: str) -> str:
    r"""clean_formula from run_dpsk_ocr2_eval_batch.py: drop "\quad (n)" equation tags inside \[...\]."""
    def one(m):
        formula = re.sub(r"\\quad\s*\([^)]*\)", "", m.group(1)).strip()
        return r"\[" + formula + r"\]"
    return re.sub(r"\\\[(.*?)\\\]", one, text)


def deepseek_ocr2_postprocess(raw: str) -> str:
    content = _ds_clean_formula((raw or "").replace(_DS_EOS, ""))   # EOS: only if an engine leaves it in
    for span, _, _ in _DS_REF_DET.findall(content):
        content = content.replace(span, "").replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n")
    return content


# --------------------------------------------------------------------------------------------------------
# The registry
#   repo                  default weights (override with --ckpt, e.g. a fine-tuned copy)
#   prompt                page-parsing prompt (ThaiOCRBench uses each item's own question instead)
#   preprocess            PIL -> PIL before the request (resizing done by the processor is not here)
#   postprocess(raw)      model output -> Markdown; postprocess_ctx(raw, origin, image) when it needs the page
#   mm_processor_kwargs   vLLM image processor limits (None = the model's own)
#   llm_extra             extra vllm.LLM(...) arguments
#   runner                a model that is a multi-stage pipeline instead of one prompt (PaddleOCR-VL)
#   vllm_env              environment variables set before vLLM starts (e.g. which model runner)
#   raw_prompt            True: no chat template; llm.generate on text_prefix + prompt (OpenThai / Parse)
#   tokenization_kwargs   raw_prompt only: how vLLM tokenizes the prompt (default: adds BOS, as DeepSeek-OCR-2 needs)
#   sampling_extra        extra SamplingParams (top_k, repetition_penalty, skip_special_tokens, ...)
#   setup(model)          called once with the weights path before inference
#   not_applicable        benchmarks the model cannot run (no free-form questions -> no ThaiOCRBench)
# --------------------------------------------------------------------------------------------------------
_TYPHOON_OCR2 = {
    "prompt": DOC2JSON_PROMPT,
    "preprocess": lambda im: im,            # the processor resizes (mm_processor_kwargs)
    "postprocess": doc2json_to_markdown,
    "max_tokens": 16384,
    "max_model_len": 30000,
    "temperature": 0.0, "top_p": 1.0, "stop": None,
    "mm_processor_kwargs": {"min_pixels": 2048, "max_pixels": 16777216},   # up to 16.7 MP
    "trust_remote_code": True,
    "chat_template_kwargs": {"enable_thinking": False},
    "llm_extra": {"enable_prefix_caching": False, "gpu_memory_utilization": 0.9},
}

_NEMOTRON_PARSE = {
    "prompt": NEMOTRON_PARSE_PROMPT,
    "raw_prompt": True,
    "text_prefix": "<pad>",              # the image placeholder, no newline after it
    "tokenization_kwargs": {"add_special_tokens": False},   # docs/serving.md: no BOS/EOS added to the prompt
    "setup": _nemotron_parse_setup,
    "preprocess": lambda im: im,         # the processor resizes
    "postprocess": nemotron_parse_postprocess,
    "max_tokens": 8744, "max_model_len": 16384,   # 3,329 image tokens + prompt + 8,744 new tokens
    "temperature": 0.2, "top_p": 1.0, "seed": 0, "stop": None,
    "sampling_extra": {"top_k": 50, "repetition_penalty": 1.05, "skip_special_tokens": False},
    "mm_processor_kwargs": None,
    "trust_remote_code": True,
    "llm_extra": {"enable_prefix_caching": False, "gpu_memory_utilization": 0.7},
    "not_applicable": {"thai"},          # fixed task prompts only, no free-form questions
}

MODELS = {
    "typhoon-ocr2-2b": {**_TYPHOON_OCR2, "repo": "typhoon-ai/typhoon-ocr2-2b"},
    "typhoon-ocr2-0.8b": {**_TYPHOON_OCR2, "repo": "typhoon-ai/typhoon-ocr2-0.8b"},
    "infinity-parser2-flash": {
        **_TYPHOON_OCR2,
        "repo": "infly/Infinity-Parser2-Flash",
        "max_model_len": 65536,
        "llm_extra": {"enable_prefix_caching": False, "gpu_memory_utilization": 0.7},
    },
    "typhoon-ocr1.5-2b": {
        "repo": "typhoon-ai/typhoon-ocr1.5-2b",
        "prompt": TYPHOON15_PROMPT,
        "preprocess": lambda im: resize_longest(im, 1800),
        "postprocess": lambda t: (t or "").strip(),
        "max_tokens": 16384, "max_model_len": 49152,
        "temperature": 0.0, "top_p": 1.0, "stop": None,
        "mm_processor_kwargs": None,
        "llm_extra": {"enable_prefix_caching": False, "gpu_memory_utilization": 0.7},
    },
    "chandra-ocr-2": {
        "repo": "datalab-to/chandra-ocr-2",
        "prompt": lambda: _chandra()[2]["ocr_layout"],   # resolved when the model is loaded
        "preprocess": chandra_preprocess,
        "postprocess": chandra_postprocess,
        "max_tokens": 16384, "max_model_len": 49152,
        "temperature": 0.0, "top_p": 0.1, "stop": ["<|im_end|>"],
        "mm_processor_kwargs": {"min_pixels": 3136, "max_pixels": 6291456},
        "llm_extra": {"enable_prefix_caching": False, "gpu_memory_utilization": 0.7},
    },
    "dots.mocr": {
        "repo": "dots-studio/dots.mocr",
        "prompt": DOTS_PROMPT_LAYOUT_ALL,
        "text_prefix": DOTS_IMG_PREFIX,
        "vqa_system_prompt": "You are a helpful assistant.",   # parser.py free-form question path
        "chat_template_content_format": "string",
        "preprocess": dots_preprocess,
        "postprocess_ctx": dots_postprocess,
        "max_tokens": 16384, "max_model_len": 30000,
        "temperature": 0.1, "top_p": 1.0, "seed": 0,           # seed: our addition, makes T=0.1 repeatable
        "stop": None,
        "mm_processor_kwargs": None,                           # resized client-side
        "trust_remote_code": True,
        "llm_extra": {"enable_prefix_caching": False, "gpu_memory_utilization": 0.7},
    },
    "paddleocr-vl-1.6": {
        "repo": "PaddlePaddle/PaddleOCR-VL-1.6",
        "runner": "paddleocr_vl",           # ocr_eval/paddle_vl.py
        "max_model_len": 30000,
        "trust_remote_code": True,
        "llm_extra": {"gpu_memory_utilization": 0.7},   # leaves room for the layout model on the same GPU
        "not_applicable": {"thai"},         # fixed task prompts only, no free-form questions
    },
    "ovisocr2": {
        "repo": "ATH-MaaS/OvisOCR2",
        "prompt": OVIS_OCR2_PROMPT,
        "preprocess": lambda im: im,
        "postprocess": ovis_ocr2_postprocess,
        "max_tokens": 16384, "max_model_len": 30000,
        "temperature": 0.0, "top_p": 1.0, "stop": None,
        "mm_processor_kwargs": {"min_pixels": 448 * 448, "max_pixels": 2880 * 2880},
        "chat_template_kwargs": {"enable_thinking": False},
        "llm_extra": {"enable_prefix_caching": False, "gpu_memory_utilization": 0.7,
                      "gdn_prefill_backend": "triton"},
    },
    "openthai2.0-ocr-markdown": {**_NEMOTRON_PARSE, "repo": OPENTHAI_REPO},
    "nemotron-3.5-parse": {**_NEMOTRON_PARSE, "repo": NEMOTRON_PARSE_REPO},
    "deepseek-ocr-2": {
        "repo": "deepseek-ai/DeepSeek-OCR-2",
        "prompt": DEEPSEEK_OCR2_PROMPT,
        "raw_prompt": True,
        "text_prefix": "<image>\n",           # ThaiOCRBench: "<image>\n" + the item's question
        "preprocess": lambda im: im,          # vLLM's processor tiles the page
        "postprocess": deepseek_ocr2_postprocess,
        "max_tokens": 8192, "max_model_len": 8192,
        "temperature": 0.0, "top_p": 1.0, "stop": None,
        "sampling_extra": {"skip_special_tokens": False,
                           "extra_args": {"ngram_size": 40, "window_size": 90,
                                          "whitelist_token_ids": {128821, 128822}}},   # <td>, </td>
        "mm_processor_kwargs": None,
        "trust_remote_code": True,
        "llm_extra": {"enable_prefix_caching": False, "mm_processor_cache_gb": 0, "gpu_memory_utilization": 0.7,
                      "logits_processors": ["vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"]},
        # VLLM_USE_V2_MODEL_RUNNER=0: vLLM's n-gram processor is written for the V1 model runner; vLLM 0.33
        # defaults to Model Runner V2, which only takes processors of its own interface (no-op on older vLLM).
        # TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1: vLLM 0.33's fused DeepEncoder attention kernel reads LOG2E, a plain
        # float in vllm.triton_utils, which Triton 3.8 rejects; this lets Triton inline it (same math).
        # VLLM_USE_FLASHINFER_SAMPLER=0: the V1 runner otherwise JIT-builds FlashInfer's top-k/top-p sampler at
        # startup (needs a host C++ compiler); decoding is greedy, which never uses that sampler.
        "vllm_env": {"VLLM_USE_V2_MODEL_RUNNER": "0", "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS": "1",
                     "VLLM_USE_FLASHINFER_SAMPLER": "0"},
    },
}


def model_for_repo(repo: str):
    """The registry key whose default weights are `repo`, or None."""
    for key, spec in MODELS.items():
        if spec["repo"].lower() == repo.lower():
            return key
    return None


def upscale_min_long(im: Image.Image, min_long: int) -> Image.Image:
    """Enlarge `im` so its long side is `min_long` px (aspect ratio kept); larger images pass through."""
    s = min_long / max(im.size)
    if s <= 1:
        return im
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)


def resolve(key: str, upscale: int = 0, max_pixels: int = None) -> dict:
    """A copy of MODELS[key] ready to run: the prompt resolved, and optional upscaling / pixel limit applied."""
    spec = dict(MODELS[key])
    if callable(spec.get("prompt")):
        spec["prompt"] = spec["prompt"]()
    if max_pixels:
        if spec.get("mm_processor_kwargs") is None:
            raise SystemExit(f"--max-pixels is not supported for {key}: it resizes images itself")
        spec["mm_processor_kwargs"] = {**spec["mm_processor_kwargs"], "max_pixels": max_pixels}
    if upscale:
        if spec.get("runner"):
            raise SystemExit(f"--upscale is not supported for {key}: it runs its own image pipeline")
        pre = spec["preprocess"]
        spec["preprocess"] = lambda im, _pre=pre, _L=upscale: _pre(upscale_min_long(im, _L))
    return spec
