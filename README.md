# Thai OCR evaluation

Code to reproduce the benchmark results of [Typhoon OCR V2](https://huggingface.co/typhoon-ai/typhoon-ocr2-2b): 8 OCR models on 5 document benchmarks, each model run with its own prompt, image preprocessing and output conversion, and each benchmark scored with its own official scorer.

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

These are the numbers on the [Typhoon OCR V2 model card](https://huggingface.co/typhoon-ai/typhoon-ocr2-2b#results), produced with this code. Typhoon OCR V2 and Infinity-Parser2-Flash were run with vLLM 0.27.1 in eager mode. Typhoon OCR 1.5's MDPBench scores were measured with vLLM 0.33.0; with vLLM 0.27.1 in its default mode it scored 62.8 (64.3 on Thai pages).

| Model | MDPBench ↑ | MDPBench TH ↑ | olmOCR-bench ↑ | ThaiOCRBench ↑ | SEA-DocBench NED ↓ | OmniDocBench v1.6 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Chandra OCR 2 | 74.2 | 71.1 | 84.4 | 26.4 | 0.1141 | 89.65 |
| dots.mocr | 76.2 | 73.6 | 85.2 | 37.9 | 0.1135 | 91.09 |
| PaddleOCR-VL-1.6 | 72.5 | 68.4 | 80.8 | n/a | 0.1323 | 95.83 |
| OvisOCR2 | 47.7 | 14.7 | 72.3 | 10.2 | 0.3843 | 96.34 |
| Infinity-Parser2-Flash | 68.3 | 61.3 | 84.4 | 35.0 | 0.1266 | 91.03 |
| Typhoon OCR 1.5 | 66.0 | 68.9 | 62.0 | 40.5 | 0.1726 | 70.97 |
| Typhoon OCR V2 0.8B | 77.4 | 77.2 | 81.7 | 36.4 | 0.1081 | 89.37 |
| Typhoon OCR V2 2B | 79.8 | 81.3 | 84.7 | 59.9 | 0.1153 | 90.97 |

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
| inference (`--python`) | running the models | `pip install -r requirements.txt` (vLLM 0.30 or newer, see below) |
| scoring (`--score-python`) | MDPBench, olmOCR-bench, ThaiOCRBench, SEA-DocBench | each benchmark's `requirements.txt`, `pip install -e "benchmarks/olmocr[bench]"` and `playwright install chromium` |
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
```

| Option | Meaning |
|---|---|
| `--model`, `--ckpt` | the recipe and the weights; `--ckpt` alone works for the models' own Hugging Face ids |
| `--benchmarks` | any of `mdp olmocr thai sea omni` (default: all, in that order) |
| `--gpu`, `--gpu-mem`, `--tp` | GPU(s), vLLM memory share, tensor parallel size |
| `--enforce-eager` | vLLM eager mode (needed on vLLM < 0.30 for the Qwen3.5-based models) |
| `--upscale PX` | enlarge images whose long side is under PX px before inference (e.g. 1600; raises ThaiOCRBench for Typhoon OCR V2) |
| `--name` | run name (default: the model key, plus `__eager`, `__up1600`, ... for non-default settings) |
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
