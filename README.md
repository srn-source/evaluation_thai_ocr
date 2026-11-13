# Thai OCR evaluation

Code to reproduce the benchmark results of [Typhoon OCR V2](https://huggingface.co/typhoon-ai/typhoon-ocr2-2b): 11 OCR models on 5 document benchmarks, each model run with its own prompt, image preprocessing and output conversion, and each benchmark scored with its own official scorer.

```bash
python run_all_benchmarks.py --ckpt typhoon-ai/typhoon-ocr2-2b
```

## Models

| Model | `--model` | Weights | Notes |
|---|---|---|---|
| Typhoon OCR V2 2B | `typhoon-ocr2-2b` | [typhoon-ai/typhoon-ocr2-2b](https://huggingface.co/typhoon-ai/typhoon-ocr2-2b) | layout JSON → Markdown; headers and footers dropped |
| Typhoon OCR V2 0.8B | `typhoon-ocr2-0.8b` | [typhoon-ai/typhoon-ocr2-0.8b](https://huggingface.co/typhoon-ai/typhoon-ocr2-0.8b) | same as the 2B |
| Infinity-Parser2-Flash | `infinity-parser2-flash` | [infly/Infinity-Parser2-Flash](https://huggingface.co/infly/Infinity-Parser2-Flash) | same prompt and conversion as Typhoon OCR V2 |
| Typhoon OCR 1.5 | `typhoon-ocr1.5-2b` | [typhoon-ai/typhoon-ocr1.5-2b](https://huggingface.co/typhoon-ai/typhoon-ocr1.5-2b) | Markdown prompt; pages resized to at most 1,800 px |
| Chandra OCR 2 | `chandra-ocr-2` | [datalab-to/chandra-ocr-2](https://huggingface.co/datalab-to/chandra-ocr-2) | prompt, resizing and HTML → Markdown from the `chandra-ocr` package |
| dots.mocr | `dots.mocr` | [dots-studio/dots.mocr](https://huggingface.co/dots-studio/dots.mocr) | preprocessing and conversion imported from the [dots.mocr](https://github.com/rednote-hilab/dots.mocr) repo |
| PaddleOCR-VL-1.6 | `paddleocr-vl-1.6` | [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | official two-stage pipeline (layout model + VLM); no ThaiOCRBench (no free-form questions) |
| OvisOCR2 | `ovisocr2` | [ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | prompt and post-processing from its model card |
| OpenThai2.0-OCR-Markdown | `openthai2.0-ocr-markdown` | [iapp/OpenThai2.0-OCR-Markdown](https://huggingface.co/iapp/OpenThai2.0-OCR-Markdown) | Nemotron 3.5 Parse fine-tune, run as its card says (NVIDIA's `DocumentParser`, single pass; postprocessing imported from the model repo); needs vLLM 0.33 or newer; no ThaiOCRBench (no free-form questions) |
| Nemotron 3.5 Parse | `nemotron-3.5-parse` | [nvidia/Nemotron-3.5-Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse) | the base model of OpenThai2.0-OCR-Markdown; same recipe (the two repos differ only in their weights) |
| DeepSeek-OCR-2 | `deepseek-ocr-2` | [deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | prompt, decoding (with its no-repeat n-gram logits processor) and post-processing from its official OmniDocBench script |

Everything that defines how a model is run is in [ocr_eval/models.py](ocr_eval/models.py). All benchmarks use the same entry, so a model gets the same prompt, image size and decoding on every benchmark. To evaluate your own fine-tune, pick the recipe with `--model` and the weights with `--ckpt`.

## Benchmarks

| `--benchmarks` | Benchmark | Pages | Score |
|---|---|---|---|
| `mdp` | [MDPBench](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/MDPBench) | 320 English and Thai pages of the public set | Overall (0–100), from the benchmark's `calculate_scores.py` |
| `olmocr` | [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) | 1,403 PDFs (page 1 rendered at 2,048 px) | average pass rate (%) |
| `thai` | [ThaiOCRBench](https://huggingface.co/datasets/scb10x/ThaiOCRBench) | 2,808 questions | mean score × 100 over the 13 tasks |
| `sea` | [SEA-DocBench](https://github.com/Shopee-MUG/SEA-Vision) | 3,091 English and Thai pages | NED, lower is better: mean of the text, table, formula and reading-order NEDs, averaged over the two languages |
| `omni` | [OmniDocBench v1.6](https://github.com/opendatalab/OmniDocBench) | all 1,651 pages | Overall = ((1 − text edit) × 100 + table TEDS + formula CDM) / 3 |

## Results

These are the numbers on the [Typhoon OCR V2 model card](https://huggingface.co/typhoon-ai/typhoon-ocr2-2b#results), produced with this code. Typhoon OCR V2 and Infinity-Parser2-Flash were run with vLLM 0.27.1 in eager mode. Typhoon OCR 1.5's MDPBench scores were measured with vLLM 0.33.0; with vLLM 0.27.1 in its default mode it scored 62.8 (64.3 on Thai pages). OpenThai2.0-OCR-Markdown and Nemotron 3.5 Parse were run with vLLM 0.33.0. For both, two things in the output are normalized without changing the content: the `rowspan="1"` and `colspan="1"` attributes that NVIDIA's post-processing puts on every table cell are removed (the table edit distance counts them as errors), and the Thai vowel sara am is written as U+0E33 (the tokenizer can only produce U+0E4D U+0E32). DeepSeek-OCR-2 was run with vLLM 0.33.0 (V1 model runner, for its no-repeat n-gram logits processor); on photographed pages it often repeats itself until the token limit.

| Model | MDPBench ↑ | MDPBench TH ↑ | olmOCR-bench ↑ | ThaiOCRBench ↑ | SEA-DocBench NED ↓ | OmniDocBench v1.6 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Chandra OCR 2 | 74.2 | 71.1 | 84.4 | 26.4 | 0.1141 | 89.65 |
| dots.mocr | 76.2 | 73.6 | 85.2 | 37.9 | 0.1135 | 91.09 |
| PaddleOCR-VL-1.6 | 72.5 | 68.4 | 80.8 | n/a | 0.1323 | 95.83 |
| OvisOCR2 | 47.7 | 14.7 | 72.3 | 10.2 | 0.3843 | 96.34 |
| Infinity-Parser2-Flash | 68.3 | 61.3 | 84.4 | 35.0 | 0.1266 | 91.03 |
| Nemotron 3.5 Parse | 74.7 | 75.9 | 62.7 | n/a | 0.1046† | 80.13 |
| OpenThai2.0-OCR-Markdown | 68.6 | 69.7 | 51.2 | n/a | 0.3342 | 62.02 |
| DeepSeek-OCR-2 | 48.9 | 40.7 | 78.2 | 10.1 | 0.1356 | 90.05 |
| Typhoon OCR 1.5 | 66.0 | 68.9 | 62.0 | 40.5 | 0.1726 | 70.97 |
| Typhoon OCR V2 0.8B | 77.4 | 77.2 | 81.7 | 36.4 | 0.1081 | 89.37 |
| Typhoon OCR V2 2B | 79.8 | 81.3 | 84.7 | 59.9 | 0.1153 | 90.97 |

† Nemotron 3.5 Parse's SEA-DocBench NED is not directly comparable. It writes most display formulas inline, and on the Thai pages the scorer counted all of them as text, so its Thai NED averages three parts instead of four. Counting only text, tables and reading order for every model, it scores 0.075, close to the best (Typhoon OCR V2 2B and dots.mocr, both 0.074).

**Reading OpenThai2.0-OCR-Markdown's scores.** Its low olmOCR-bench, SEA-DocBench and OmniDocBench scores come mostly from formulas and from page headers and footers, not from how well it reads text and tables:

- It writes formulas as plain text, and these benchmarks only accept LaTeX. It scores 2.3 and 0.0 on the two olmOCR-bench math categories, 24.6 on OmniDocBench's formula metric (CDM) and about 1.0 (the worst value) on SEA-DocBench's formula NED.
- SEA-DocBench averages four parts with equal weight, so the formula part alone moves its NED from 0.114 (text, tables and reading order) to 0.334.
- It keeps page headers and footers, as its official pipeline returns them. olmOCR-bench expects them to be left out, so it scores 23.4 on that category. On the other five olmOCR-bench categories it averages 76.8.

Expect small differences between runs (a few tenths of a point on MDPBench), even with greedy decoding: vLLM's batching and its default compiled mode change some outputs slightly.

## Setup

### 1. Benchmarks and data

```bash
bash setup_benchmarks.sh
```

This clones the benchmarks (and dots.mocr) into `benchmarks/` at the commits we used, applies three small fixes to their scorers (in [patches/](patches)), and downloads the data:

- `mdpbench.patch` lets the config select several languages, so English and Thai are scored together.
- `thaiocrbench.patch` and `omnidocbench.patch` stop the scorers from crashing or hanging on very long, looping outputs. They do not change any score.

SEA-DocBench's ground truth (`ground_truth.json`) is not on Hugging Face yet; ask the [SEA-Vision](https://github.com/Shopee-MUG/SEA-Vision) authors for it. If you already have clones, point to them instead: `MDPBENCH_DIR`, `OLMOCR_DIR`, `THAIOCRBENCH_DIR`, `SEA_DIR`, `OMNI_DIR` and `DOTS_MOCR_REPO`.

### 2. Python environments

The benchmarks' scorers have conflicting dependencies, so we use separate environments. Pass their Python executables with `--python`, `--score-python` and `--omni-python`.

| Environment | Used for | Install |
|---|---|---|
| inference (`--python`) | running the models | `pip install -r requirements.txt` (vLLM 0.30 or newer, see below; 0.33 or newer for OpenThai2.0-OCR-Markdown) |
| scoring (`--score-python`) | MDPBench, olmOCR-bench, ThaiOCRBench, SEA-DocBench | each benchmark's `requirements.txt`, `pip install -e "benchmarks/olmocr[bench]"` and `playwright install --with-deps chromium` (olmOCR-bench renders its math tests in headless Chromium; without root, install only the browser and point `LD_LIBRARY_PATH` at its system libraries) |
| OmniDocBench (`--omni-python`) | OmniDocBench | Python 3.10 with OmniDocBench's dependencies; the formula metric (CDM) also needs TeX Live, ImageMagick and Ghostscript (put TeX Live's `bin` in `TEXLIVE_BIN`) |
| PaddlePaddle (`PADDLE_PYTHON`) | PaddleOCR-VL-1.6 only | `pip install paddlepaddle-gpu` ([install guide](https://www.paddlepaddle.org.cn/install/quick)) and `pip install "paddleocr[doc-parser]>=3.6.0"` |

The inference environment also needs poppler (`pdftoppm`, `pdfinfo`; e.g. `apt install poppler-utils` or `conda install -c conda-forge poppler`) to render olmOCR-bench PDFs, `pip install chandra-ocr` for Chandra OCR 2, and `PyMuPDF` for dots.mocr.

**vLLM version.** Typhoon OCR V2, Infinity-Parser2 and OvisOCR2 are based on Qwen3.5. With vLLM older than 0.30, vLLM's default compiled mode lowers their accuracy (MDPBench 70.8 instead of 79.8 for Typhoon OCR V2 2B on vLLM 0.27.1); add `--enforce-eager` there. With vLLM 0.30.0 and 0.33.0 the default mode scored 79.6 and 79.7.

## Usage

```bash
# one model, all five benchmarks
python run_all_benchmarks.py --ckpt typhoon-ai/typhoon-ocr2-2b \
  --python /path/to/infer-env/bin/python --score-python /path/to/score-env/bin/python \
  --omni-python /path/to/omni-env/bin/python

# some benchmarks only, on GPU 1, sharing the GPU
python run_all_benchmarks.py --model chandra-ocr-2 --benchmarks mdp thai --gpu 1 --gpu-mem 0.75

# your own fine-tune with the Typhoon OCR V2 recipe
python run_all_benchmarks.py --model typhoon-ocr2-2b --ckpt /path/to/checkpoint --name my_finetune

# Typhoon OCR V2 with its EAGLE-3 draft head (speculative decoding; run name typhoon-ocr2-2b__eagle3k3)
python run_all_benchmarks.py --ckpt typhoon-ai/typhoon-ocr2-2b --benchmarks mdp \
  --spec-model typhoon-ai/typhoon-ocr2-2b-eagle3
```

| Option | Meaning |
|---|---|
| `--model`, `--ckpt` | the recipe and the weights; `--ckpt` alone works for the models' own Hugging Face ids |
| `--benchmarks` | any of `mdp olmocr thai sea omni` (default: all, in that order) |
| `--gpu`, `--gpu-mem`, `--tp` | GPU(s), vLLM memory share, tensor parallel size |
| `--enforce-eager` | vLLM eager mode (needed on vLLM < 0.30 for the Qwen3.5-based models) |
| `--spec-model`, `--num-spec-tokens` | speculative decoding with a draft head (local dir or Hugging Face id), e.g. [typhoon-ai/typhoon-ocr2-2b-eagle3](https://huggingface.co/typhoon-ai/typhoon-ocr2-2b-eagle3) or [typhoon-ai/typhoon-ocr2-0.8b-eagle3](https://huggingface.co/typhoon-ai/typhoon-ocr2-0.8b-eagle3) (`--spec-method`, default `eagle3`; 3 draft tokens by default). Tested with vLLM 0.33.0; not available for PaddleOCR-VL. Greedy outputs can differ slightly from decoding without it, as they do between repeated vLLM runs |
| `--upscale PX` | enlarge images whose long side is under PX px before inference (e.g. 1600; raises ThaiOCRBench for Typhoon OCR V2) |
| `--name` | run name (default: the model key, plus `__eager`, `__eagle3k3`, `__up1600`, ... for non-default settings) |
| `--skip-infer`, `--rescore`, `--force` | score existing predictions only, re-score, or regenerate predictions |
| `--report-only`, `--dry-run` | rebuild `RESULTS.md`, or print the commands without running them |

Each benchmark's predictions are written inside its clone (for example `benchmarks/MultimodalOCR/MDPBench/result/<run>/`), because the scorers read them from there. Scores go to `results/<run>/<benchmark>.json`, all runs are tabled in `RESULTS.md` and `results_<benchmark>.csv`, and every step's output is logged in `logs/<run>/`.

To run one model on one benchmark without scoring:

```bash
python -m ocr_eval.infer --bench mdp --model-key typhoon-ocr2-2b \
  --data benchmarks/MultimodalOCR/MDPBench/MDPBench_dataset/MDPBench_img_public --out preds/mdp
```

## License

Apache-2.0, see [LICENSE](LICENSE). The benchmarks, datasets and models keep their own licenses; see [NOTICE](NOTICE) for the code adapted from other projects.
