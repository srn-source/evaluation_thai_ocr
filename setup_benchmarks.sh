#!/usr/bin/env bash
# Clone the five benchmarks (and dots.mocr) at the commits we used, apply our small scorer fixes, and download
# the data. Everything goes under ./benchmarks (or $BENCH_ROOT). Safe to run again: existing clones are kept.
#
#   bash setup_benchmarks.sh            # code + data
#   bash setup_benchmarks.sh --no-data  # code only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BENCH_ROOT:-$HERE/benchmarks}"
DATA=1
[[ "${1:-}" == "--no-data" ]] && DATA=0
mkdir -p "$ROOT"

clone() {  # clone <url> <dir> <commit> [patch]
  local url=$1 dir=$ROOT/$2 commit=$3 patch=${4:-}
  if [[ ! -d "$dir/.git" ]]; then
    git clone "$url" "$dir"
  fi
  git -C "$dir" checkout -q "$commit"
  if [[ -n "$patch" ]]; then
    if git -C "$dir" apply --check "$HERE/patches/$patch" 2>/dev/null; then
      git -C "$dir" apply "$HERE/patches/$patch"
      echo "applied patches/$patch to $2"
    else
      echo "patches/$patch already applied to $2 (or does not apply; check git -C $dir diff)"
    fi
  fi
}

# MDPBench: the patch lets the config filter on a list of languages (English + Thai)
clone https://github.com/Yuliang-Liu/MultimodalOCR.git MultimodalOCR 1e8061fefb82b56fbad0930cb678f738f5dea902 mdpbench.patch
clone https://github.com/allenai/olmocr.git olmocr f7cfe4c22098b154c76b6ec950d1c0a464eecf8d
# ThaiOCRBench: the patch stops the scorer crashing on very long (looping) answers; the scores are unchanged
clone https://github.com/scb-10x/ThaiOCRBench.git ThaiOCRBench c6f3959265c8e11045b16aeb1e88e5e1e30b6167 thaiocrbench.patch
clone https://github.com/Shopee-MUG/SEA-Vision.git SEA-Vision e2da8a89ca5813761a4092d85750d16417fbecbb
# OmniDocBench: the patch fixes a deadlock and a recursion-limit crash in the matcher; the scores are unchanged
clone https://github.com/opendatalab/OmniDocBench.git OmniDocBench 193627ae9e97d89188468ed1ee3b7a856ff76044 omnidocbench.patch
clone https://github.com/rednote-hilab/dots.mocr.git dots.mocr 23f3e5612fb8066d4034d5ecfc8f33a9243533eb

[[ $DATA == 0 ]] && { echo "Code ready under $ROOT (data skipped)."; exit 0; }

# Data (needs: pip install -U huggingface_hub)
echo "== MDPBench (public set)"
(cd "$ROOT/MultimodalOCR/MDPBench" && python tools/download_dataset.py)

echo "== olmOCR-bench"
hf download allenai/olmOCR-bench --repo-type dataset --local-dir "$ROOT/olmocr/olmOCR-bench"

echo "== OmniDocBench v1.6 (revision we used)"
hf download opendatalab/OmniDocBench --repo-type dataset --revision aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec \
  --local-dir "$ROOT/OmniDocBench/OmniDocBench_dataset"

echo "== SEA-DocBench images"
SEA="$ROOT/SEA-Vision/SEA-DocBench"
mkdir -p "$SEA/data"
if [[ ! -d "$SEA/data/images" ]]; then
  hf download xingranzhao/SEA-Vision SEA-DocBench-images.tar.gz --repo-type dataset --local-dir "$SEA"
  tar xzf "$SEA/SEA-DocBench-images.tar.gz" -C "$SEA/data/"
fi
[[ -f "$SEA/data/ground_truth.json" ]] || echo "NOTE: SEA-DocBench's ground_truth.json is not on Hugging Face yet; " \
  "get it from the SEA-Vision authors and put it at $SEA/data/ground_truth.json"

echo "ThaiOCRBench (scb10x/ThaiOCRBench) is downloaded automatically at inference time."
echo "Done: $ROOT"
