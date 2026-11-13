#!/usr/bin/env python
"""Run one model on the five benchmarks (inference, then each benchmark's own scorer) and collect the scores.

    python run_all_benchmarks.py --ckpt typhoon-ai/typhoon-ocr2-2b
    python run_all_benchmarks.py --model chandra-ocr-2 --benchmarks mdp thai
    python run_all_benchmarks.py --model typhoon-ocr2-2b --ckpt /path/to/my-finetune --name my_finetune

Models (--model; see ocr_eval/models.py): typhoon-ocr2-2b, typhoon-ocr2-0.8b, infinity-parser2-flash,
typhoon-ocr1.5-2b, chandra-ocr-2, dots.mocr, paddleocr-vl-1.6, ovisocr2, openthai2.0-ocr-markdown,
nemotron-3.5-parse, deepseek-ocr-2. With only --ckpt, the model whose default weights are --ckpt is used.

Benchmarks (--benchmarks, default all, in this order):
  mdp     MDPBench, English + Thai pages      ocr_eval.infer -> pdf_validation.py -> tools/calculate_scores.py
  olmocr  olmOCR-bench                        ocr_eval.infer -> python -m olmocr.bench.benchmark
  thai    ThaiOCRBench                        ocr_eval.infer -> eval_scripts/eval.py
  sea     SEA-DocBench, English + Thai pages  ocr_eval.infer -> pdf_validation.py (once per language)
  omni    OmniDocBench v1.6, all 1651 pages   ocr_eval.infer -> pdf_validation.py (CDM needs TeX Live)

Each benchmark lives in its own clone under benchmarks/ (run setup_benchmarks.sh first, or point the
MDPBENCH_DIR, OLMOCR_DIR, THAIOCRBENCH_DIR, SEA_DIR, OMNI_DIR variables at existing clones). Predictions
are written inside those clones, as their scorers expect; scores go to results/<run>/<bench>.json and
the tables in RESULTS.md.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ocr_eval.models import MODELS, model_for_repo  # noqa: E402  (no vLLM import)

BENCH_ROOT = Path(os.environ.get("BENCH_ROOT", HERE / "benchmarks"))
MDP = Path(os.environ.get("MDPBENCH_DIR", BENCH_ROOT / "MultimodalOCR" / "MDPBench"))
OLMOCR = Path(os.environ.get("OLMOCR_DIR", BENCH_ROOT / "olmocr"))
THAI = Path(os.environ.get("THAIOCRBENCH_DIR", BENCH_ROOT / "ThaiOCRBench"))
SEA = Path(os.environ.get("SEA_DIR", BENCH_ROOT / "SEA-Vision" / "SEA-DocBench"))
OMNI = Path(os.environ.get("OMNI_DIR", BENCH_ROOT / "OmniDocBench"))
CONFIGS = HERE / "configs"
RESULTS_DIR, LOG_DIR = HERE / "results", HERE / "logs"
THAI_N_ITEMS = 2808   # size of the scb10x/ThaiOCRBench test split

ALL_BENCHES = ["mdp", "olmocr", "thai", "sea", "omni"]
TITLES = {"mdp": "MDPBench (en+th)", "olmocr": "olmOCR-bench", "thai": "ThaiOCRBench",
          "sea": "SEA-DocBench (en+th)", "omni": "OmniDocBench v1.6"}


class Run:
    def __init__(self, args):
        self.args = args
        self.key = args.model
        self.weights = args.ckpt or MODELS[self.key]["repo"]
        if args.name:
            self.name = args.name
        else:
            self.name = self.key
            if args.ckpt and args.ckpt.rstrip("/").lower() != MODELS[self.key]["repo"].lower():
                self.name += "__" + re.sub(r"[^A-Za-z0-9_.-]+", "_", args.ckpt.strip("/").replace("/", "__"))
            if args.max_pixels:
                self.name += f"__maxpx{args.max_pixels}"
            if args.upscale:
                self.name += f"__up{args.upscale}"
            if args.enforce_eager:
                self.name += "__eager"
            if args.spec_model:
                self.name += f"__{args.spec_method}k{args.num_spec_tokens}"
        self.res_dir, self.log_dir = RESULTS_DIR / self.name, LOG_DIR / self.name
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                        PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        if args.hf_home:
            self.env["HF_HOME"] = args.hf_home
        # vLLM builds kernels with ninja from the env's bin dir: put it on PATH even without `conda activate`
        self.env["PATH"] = f"{Path(args.python).parent}:{self.env.get('PATH', '')}"

    def infer_cmd(self, bench, out, data=None, extra=()):
        a = self.args
        cmd = [a.python, "-m", "ocr_eval.infer", "--bench", bench, "--model-key", self.key, "--model", self.weights,
               "--out", str(out), *extra]
        if data:
            cmd += ["--data", str(data)]
        if a.gpu_mem:
            cmd += ["--gpu-memory-utilization", str(a.gpu_mem)]
        if a.enforce_eager:
            cmd.append("--enforce-eager")
        if a.spec_model:
            spec_model = os.path.abspath(a.spec_model) if os.path.isdir(a.spec_model) else a.spec_model
            cmd += ["--spec-model", spec_model, "--spec-method", a.spec_method,
                    "--num-spec-tokens", str(a.num_spec_tokens)]
        if a.upscale:
            cmd += ["--upscale", str(a.upscale)]
        if a.max_pixels:
            cmd += ["--max-pixels", str(a.max_pixels)]
        if a.tp > 1:
            cmd += ["--tensor-parallel-size", str(a.tp)]
        if a.force:
            cmd.append("--force")
        return cmd

    def run(self, cmd, cwd, log_name, python=None, extra_env=None):
        """Run a step, stream its output to the terminal and logs/<run>/<log_name>.log; return (rc, output)."""
        env = dict(self.env)
        if python:  # scoring envs: their own bin dir first
            env["PATH"] = f"{Path(python).parent}:{env['PATH']}"
        env.update(extra_env or {})
        shown = f"(cd {cwd} && CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {shlex.join(map(str, cmd))})"
        print(f"\n\033[1;34m[{dt.datetime.now():%H:%M:%S}] $ {shown}\033[0m", flush=True)
        if self.args.dry_run:
            return 0, ""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path, chunks, t0 = self.log_dir / f"{log_name}.log", [], time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(shown + "\n\n")
            proc = subprocess.Popen(list(map(str, cmd)), cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                    bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                chunks.append(line)
            proc.wait()
            log.write(f"\n[exit {proc.returncode}] {time.time() - t0:.0f}s\n")
        print(f"\033[1;34m[{dt.datetime.now():%H:%M:%S}] exit={proc.returncode} "
              f"({(time.time() - t0) / 60:.1f} min) log={log_path}\033[0m", flush=True)
        return proc.returncode, "".join(chunks)

    def check(self, rc, what):
        if rc:
            raise RuntimeError(f"{what} failed (exit {rc}); see {self.log_dir}")

    def record(self, bench, metrics, paths):
        a = self.args
        return {"run": self.name, "bench": bench, "model": self.key, "weights": self.weights,
                "enforce_eager": a.enforce_eager, "upscale": a.upscale, "max_pixels": a.max_pixels,
                **({"spec_model": a.spec_model, "spec_method": a.spec_method, "num_spec_tokens": a.num_spec_tokens}
                   if a.spec_model else {}),
                "metrics": metrics, "paths": {k: str(v) for k, v in paths.items()},
                "finished_at": dt.datetime.now().isoformat(timespec="seconds")}


def config_for_run(template: Path, out: Path, data_path: str, language=None, dry_run=False):
    """Copy a config template with prediction.data_path (and filter.language) set for this run."""
    text = template.read_text(encoding="utf-8")
    text, n = re.subn(r"(prediction:\s*\n(?:[ \t]*#.*\n)*[ \t]*data_path:[ \t]*)(\S+)",
                      lambda m: m.group(1) + data_path, text, count=1)
    if n != 1:
        raise RuntimeError(f"no prediction.data_path in {template}")
    if language is not None:
        text, n = re.subn(r"(filter:\s*\n[ \t]*language:[ \t]*)(\S+)", lambda m: m.group(1) + language, text, count=1)
        if n != 1:
            raise RuntimeError(f"no filter.language in {template}")
    if not dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return out


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _num(x):
    """float or None (the scorers write "NaN" for components without samples)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _edit(m, key):
    return _num(m.get(key, {}).get("all", {}).get("Edit_dist", {}).get("ALL_page_avg"))


# --------------------------------------------------------------------------------------------------- MDPBench
def bench_mdp(r: Run):
    pred, result = MDP / "result" / r.name, MDP / "result" / f"{r.name}_result"
    if not r.args.skip_infer:
        r.check(r.run(r.infer_cmd("mdp", pred, MDP / "MDPBench_dataset" / "MDPBench_img_public"), HERE,
                      "mdp_1_infer")[0], "MDPBench inference")
    metric_json = result / f"{r.name}_result_metric_result.json"
    if r.args.rescore or not metric_json.exists():
        cfg = config_for_run(CONFIGS / "mdpbench_end2end_en_th.yaml", MDP / "configs" / "generated" / f"{r.name}.yaml",
                             f"result/{r.name}", dry_run=r.args.dry_run)
        r.check(r.run([r.args.score_python, "pdf_validation.py", "--config", cfg.relative_to(MDP)], MDP,
                      "mdp_2_validate", python=r.args.score_python)[0], "MDPBench pdf_validation")
    rc, out = r.run([r.args.score_python, "tools/calculate_scores.py", "--result_folder", f"result/{r.name}_result"],
                    MDP, "mdp_3_scores", python=r.args.score_python)
    r.check(rc, "MDPBench calculate_scores")
    if r.args.dry_run:
        return None
    metrics, section = {}, None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s.strip("[]")
            continue
        m = re.match(r"([A-Za-z\-\.]+)\s*:\s*([\d.]+|N/A)$", s)
        if m and section and m.group(2) != "N/A":
            key, val = m.group(1), float(m.group(2))
            if section == "Overall":
                metrics[f"overall_{key.rstrip('.').lower()}"] = val
            elif key == "Avg":
                metrics[f"{section.lower().replace('-', '_')}_avg"] = val
            else:
                metrics[f"lang_{key}"] = val
    m = load_json(metric_json)
    metrics.update({
        "text_edit": round(_edit(m, "text_block"), 4), "formula_edit": round(_edit(m, "display_formula"), 4),
        "table_teds": round(m["table"]["all"]["TEDS"]["all"], 4), "table_edit": round(_edit(m, "table"), 4),
        "reading_edit": round(_edit(m, "reading_order"), 4),
    })
    return r.record("mdp", metrics, {"pred": pred, "result": result})


# ----------------------------------------------------------------------------------------------- olmOCR-bench
def bench_olmocr(r: Run):
    data = OLMOCR / "olmOCR-bench" / "bench_data"
    if not r.args.skip_infer:
        r.check(r.run(r.infer_cmd("olmocr", data / r.name, data, ["--olmocr-repo", OLMOCR]), HERE,
                      "olmocr_1_infer")[0], "olmOCR-bench inference")
    rc, out = r.run([r.args.score_python, "-m", "olmocr.bench.benchmark", "--dir", "./olmOCR-bench/bench_data",
                     "--candidate", r.name], OLMOCR, "olmocr_2_score", python=r.args.score_python)
    r.check(rc, "olmOCR-bench scoring")
    if r.args.dry_run:
        return None
    final = out.split("Final Summary")[-1]
    m = re.search(r"Average Score:\s*([\d.]+)%\s*±\s*([\d.]+)%", final)
    if not m:
        if "FAILED (errors)" in final:
            raise RuntimeError("olmOCR-bench tests raised errors, so it gives no score; see the first [ERROR] in "
                               f"{r.log_dir / 'olmocr_2_score.log'} (often: headless Chromium for the math tests "
                               "cannot start; run `playwright install --with-deps chromium` or set LD_LIBRARY_PATH)")
        raise RuntimeError("could not find 'Average Score' in the olmOCR-bench output")
    metrics = {"overall": float(m.group(1)), "overall_ci": float(m.group(2))}
    for mj in re.finditer(r"^\s+(\S+?)(?:\.jsonl)?\s*:\s*([\d.]+)% \((\d+)/(\d+) tests\)", final, re.M):
        metrics[f"cat_{mj.group(1)}"] = float(mj.group(2))
    return r.record("olmocr", metrics, {"pred": data / r.name})


# ----------------------------------------------------------------------------------------------- ThaiOCRBench
def bench_thai(r: Run):
    pred, res = THAI / "pred_folder" / f"{r.name}.json", THAI / "res_folder" / f"{r.name}.json"

    def n_pred():
        try:
            return len(load_json(pred))
        except Exception:
            return 0
    complete = pred.exists() and n_pred() >= THAI_N_ITEMS   # predictions are written per batch; resume partial files
    if not r.args.skip_infer and (r.args.force or not complete):
        r.check(r.run(r.infer_cmd("thai", pred), HERE, "thai_1_infer")[0], "ThaiOCRBench inference")
    if r.args.rescore or not res.exists() or (pred.exists() and pred.stat().st_mtime > res.stat().st_mtime):
        r.check(r.run([r.args.score_python, "./eval_scripts/eval.py", "--input_path", f"./pred_folder/{r.name}.json",
                       "--output_path", f"./res_folder/{r.name}.json"], THAI, "thai_2_eval",
                      python=r.args.score_python)[0], "ThaiOCRBench eval")
    if r.args.dry_run:
        return None
    by_task = {}
    for it in load_json(res):
        if "score" in it:
            by_task.setdefault(it["type"], []).append(float(it["score"]))
    metrics = {f"task_{t}": round(100 * statistics.fmean(s), 2) for t, s in sorted(by_task.items())}
    task_means = [statistics.fmean(s) for s in by_task.values()]
    metrics["avg_task"] = round(100 * statistics.fmean(task_means), 2) if task_means else None
    metrics["n_items"] = sum(len(s) for s in by_task.values())
    return r.record("thai", metrics, {"pred": pred, "result": res})


# --------------------------------------------------------------------------------------------- SEA-DocBench
def bench_sea(r: Run):
    """Headline metric as in the SEA-DocBench paper (arXiv:2603.15409): per language, the mean of the text,
    table, formula and reading-order NEDs (lower is better); components without samples are left out."""
    out = SEA / "outputs" / r.name
    langs = ["en", "th"]
    if not r.args.skip_infer:
        r.check(r.run(r.infer_cmd("sea", out, SEA / "data" / "images",
                                  ["--gt-json", SEA / "data" / "ground_truth.json", "--langs", *langs]), HERE,
                      "sea_1_infer")[0], "SEA-DocBench inference")
    metrics = {}
    for lang in langs:
        # the scorer names its results after the prediction folder, so each language gets its own symlink
        link = SEA / "outputs" / f"{r.name}_{lang}"
        if not r.args.dry_run:
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                raise RuntimeError(f"{link} exists and is not a symlink")
            link.symlink_to(r.name)
        metric_json = SEA / "result" / f"{r.name}_{lang}_quick_match_metric_result.json"
        if r.args.rescore or not metric_json.exists():
            cfg = config_for_run(CONFIGS / "sea_end2end.yaml", SEA / "configs" / "generated" / f"{r.name}_{lang}.yaml",
                                 f"./outputs/{r.name}_{lang}", language=lang, dry_run=r.args.dry_run)
            r.check(r.run([r.args.score_python, "pdf_validation.py", "--config", cfg.relative_to(SEA)], SEA,
                          f"sea_2_validate_{lang}", python=r.args.score_python)[0], f"SEA-DocBench scoring ({lang})")
        if r.args.dry_run:
            continue
        m = load_json(metric_json)
        comps = [x for x in (_edit(m, "text_block"), _edit(m, "table"), _edit(m, "display_formula"),
                             _edit(m, "reading_order")) if x is not None]
        rd = lambda v: None if v is None else round(v, 4)
        metrics.update({f"{lang}_ned": rd(statistics.fmean(comps)) if comps else None,
                        f"{lang}_text_ned": rd(_edit(m, "text_block")), f"{lang}_table_ned": rd(_edit(m, "table")),
                        f"{lang}_table_teds": rd(_num(m.get("table", {}).get("all", {}).get("TEDS", {}).get("all"))),
                        f"{lang}_formula_ned": rd(_edit(m, "display_formula")),
                        f"{lang}_reading_ned": rd(_edit(m, "reading_order"))})
    if r.args.dry_run:
        return None
    valid = [metrics[f"{lang}_ned"] for lang in langs if metrics.get(f"{lang}_ned") is not None]
    metrics["avg_ned"] = round(statistics.fmean(valid), 4) if valid else None
    return r.record("sea", metrics, {"pred": out})


# ---------------------------------------------------------------------------------------------- OmniDocBench
def bench_omni(r: Run):
    """Leaderboard convention (OmniDocBench README): Overall = ((1 - text edit) x 100 + table TEDS + formula CDM) / 3,
    over all 1651 pages. CDM renders LaTeX: the scoring env needs TeX Live, ImageMagick and Ghostscript."""
    out = OMNI / "outputs" / r.name
    if not r.args.skip_infer:
        r.check(r.run(r.infer_cmd("omni", out, OMNI / "OmniDocBench_dataset" / "images"), HERE, "omni_1_infer")[0],
                "OmniDocBench inference")
    metric_json = OMNI / "result" / f"{r.name}_quick_match_metric_result.json"
    if r.args.rescore or not metric_json.exists():
        cfg = config_for_run(CONFIGS / "omnidocbench_end2end_full.yaml", OMNI / "configs" / "generated" / f"{r.name}.yaml",
                             f"./outputs/{r.name}", dry_run=r.args.dry_run)
        extra = {"PATH": f"{os.environ['TEXLIVE_BIN']}:{Path(r.args.omni_python).parent}:{r.env['PATH']}"} \
            if os.environ.get("TEXLIVE_BIN") else None
        r.check(r.run([r.args.omni_python, "pdf_validation.py", "--config", cfg.relative_to(OMNI)], OMNI,
                      "omni_2_validate", python=r.args.omni_python, extra_env=extra)[0], "OmniDocBench scoring")
    if r.args.dry_run:
        return None
    m = load_json(metric_json)
    pct = lambda v: None if v is None else (v * 100 if v <= 1.0 else v)   # fractions -> 0-100
    te = _edit(m, "text_block")
    teds = pct(_num(m.get("table", {}).get("all", {}).get("TEDS", {}).get("all")))
    cdm = pct(_num((m.get("display_formula", {}).get("all", {}).get("CDM") or {}).get("all")))
    overall = round(((1 - te) * 100 + teds + cdm) / 3, 2) if None not in (te, teds, cdm) else None
    rd = lambda v, nd: None if v is None else round(v, nd)
    metrics = {"overall": overall, "text_edit": rd(te, 4), "formula_cdm": rd(cdm, 2), "table_teds": rd(teds, 2),
               "reading_edit": rd(_edit(m, "reading_order"), 4)}
    return r.record("omni", metrics, {"pred": out, "result": metric_json})


BENCH_FUNCS = {"mdp": bench_mdp, "olmocr": bench_olmocr, "thai": bench_thai, "sea": bench_sea, "omni": bench_omni}

# ------------------------------------------------------------------------------------------------- report
COLUMNS = {   # (metric key, header, higher is better)
    "mdp": [("overall_all", "Overall", True), ("overall_digit", "Digital", True), ("overall_photo", "Photo", True),
            ("lang_EN", "EN", True), ("lang_TH", "TH", True), ("text_edit", "Text edit↓", False),
            ("formula_edit", "Formula edit↓", False), ("table_teds", "Table TEDS", True),
            ("table_edit", "Table edit↓", False), ("reading_edit", "Reading order edit↓", False)],
    "olmocr": [("overall", "Overall", True), ("overall_ci", "±", None), ("cat_arxiv_math", "arXiv math", True),
               ("cat_headers_footers", "Headers/footers", True), ("cat_long_tiny_text", "Long tiny text", True),
               ("cat_multi_column", "Multi-column", True), ("cat_old_scans", "Old scans", True),
               ("cat_old_scans_math", "Old scans math", True), ("cat_table_tests", "Tables", True),
               ("cat_baseline", "Baseline", True)],
    "thai": [("avg_task", "Average", True)],   # per-task columns are added from the results
    "sea": [("avg_ned", "Average NED↓", False), ("en_ned", "EN NED↓", False), ("th_ned", "TH NED↓", False),
            ("th_text_ned", "TH text↓", False), ("th_table_teds", "TH table TEDS", True)],
    "omni": [("overall", "Overall", True), ("text_edit", "Text edit↓", False), ("formula_cdm", "Formula CDM", True),
             ("table_teds", "Table TEDS", True), ("reading_edit", "Reading order edit↓", False)],
}
NOTES = {
    "mdp": "Overall, Digital, Photo, EN and TH from tools/calculate_scores.py (0-100). Edit distances and TEDS: page averages.",
    "olmocr": "Average of the per-category pass rates, as printed by olmocr.bench.benchmark.",
    "thai": "Mean score x 100 per task; Average = mean over the 13 tasks.",
    "sea": "NED per language = mean of the text, table, formula and reading-order NEDs (as in the SEA-DocBench paper).",
    "omni": "Overall = ((1 - text edit) x 100 + table TEDS + formula CDM) / 3 over all 1651 pages.",
}


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1 else f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def write_report():
    recs = {b: [] for b in ALL_BENCHES}
    for d in sorted(RESULTS_DIR.glob("*")):
        for b in ALL_BENCHES:
            if (d / f"{b}.json").exists():
                recs[b].append(load_json(d / f"{b}.json"))
    lines = ["# Results", "", f"_Written {dt.datetime.now():%Y-%m-%d %H:%M} by run_all_benchmarks.py. "
             "Bold = best in the column; ↓ = lower is better._", ""]
    for b in ALL_BENCHES:
        lines += [f"## {TITLES[b]}", ""]
        if not recs[b]:
            lines += ["_no runs yet_", ""]
            continue
        cols = list(COLUMNS[b])
        if b == "thai":
            cols += [(k, k[5:], True) for k in sorted({k for r in recs[b] for k in r["metrics"] if k.startswith("task_")})]
        best = {}
        for key, _, hib in cols:
            vals = [(r["metrics"].get(key), i) for i, r in enumerate(recs[b]) if isinstance(r["metrics"].get(key), (int, float))]
            if hib is not None and len(vals) > 1:
                best[key] = (max if hib else min)(vals)[1]
        lines.append("| run | " + " | ".join(h for _, h, _ in cols) + " |")
        lines.append("|---|" + "|".join("---:" for _ in cols) + "|")
        for i, r in enumerate(recs[b]):
            cells = [f"**{fmt(r['metrics'].get(k))}**" if best.get(k) == i and fmt(r["metrics"].get(k)) else fmt(r["metrics"].get(k))
                     for k, _, _ in cols]
            lines.append(f"| {r['run']} | " + " | ".join(cells) + " |")
        lines += ["", NOTES[b], ""]
        with open(HERE / f"results_{b}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "model", "weights", *[k for k, _, _ in cols], "finished_at"])
            for r in recs[b]:
                w.writerow([r["run"], r["model"], r["weights"], *[r["metrics"].get(k) for k, _, _ in cols], r["finished_at"]])
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {HERE / 'RESULTS.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default=None,
                    help="which model recipe to run (default: the one whose weights are --ckpt)")
    ap.add_argument("--ckpt", default=None, help="weights to load: a Hugging Face id or a local dir "
                                                 "(default: the model's own repo)")
    ap.add_argument("--benchmarks", nargs="+", choices=ALL_BENCHES, default=ALL_BENCHES)
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (default 0)")
    ap.add_argument("--gpu-mem", type=float, default=None,
                    help="vLLM gpu_memory_utilization, e.g. 0.75 on a shared GPU (default: the model's own)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="vLLM eager mode; needed for Qwen3.5-based models on vLLM < 0.30 (run name gets __eager)")
    ap.add_argument("--spec-model", default=None,
                    help="speculative decoding with this draft head (local dir or HF id), e.g. an EAGLE-3 head "
                         "(run name gets __<method>k<n>)")
    ap.add_argument("--spec-method", default="eagle3", help="vLLM speculative method for --spec-model (default eagle3)")
    ap.add_argument("--num-spec-tokens", type=int, default=3, help="draft tokens proposed per step (default 3)")
    ap.add_argument("--upscale", type=int, default=0, metavar="PX",
                    help="enlarge images under PX px (long side) to PX before inference (run name gets __up<PX>)")
    ap.add_argument("--max-pixels", type=int, default=None, help="override the image processor's max_pixels")
    ap.add_argument("--name", default=None, help="run name (default: the model key plus suffixes)")
    ap.add_argument("--python", default=sys.executable, help="python of the inference env (vLLM)")
    ap.add_argument("--score-python", default=sys.executable,
                    help="python of the scoring env for MDPBench, olmOCR-bench, ThaiOCRBench and SEA-DocBench")
    ap.add_argument("--omni-python", default=None, help="python of the OmniDocBench scoring env (default: --score-python)")
    ap.add_argument("--hf-home", default=None, help="HF_HOME for downloads (default: your environment's)")
    ap.add_argument("--skip-infer", action="store_true", help="only (re)score existing predictions")
    ap.add_argument("--rescore", action="store_true", help="re-run scoring even if results exist")
    ap.add_argument("--force", action="store_true", help="regenerate predictions that already exist")
    ap.add_argument("--report-only", action="store_true", help="only rebuild RESULTS.md from results/")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    if args.report_only:
        return write_report()
    if not args.model:
        if not args.ckpt or not model_for_repo(args.ckpt):
            ap.error("pass --model (one of: " + ", ".join(MODELS) + "), or --ckpt with a model's Hugging Face id")
        args.model = model_for_repo(args.ckpt)
    args.omni_python = args.omni_python or args.score_python
    r = Run(args)
    print(f"run       : {r.name}\nmodel     : {r.key}\nweights   : {r.weights}\ngpu       : {args.gpu}"
          f"\nbenchmarks: {' '.join(args.benchmarks)}\nresults   : {r.res_dir}")
    summary, t_all = {}, time.time()
    for b in args.benchmarks:
        print(f"\n{'=' * 78}\n {TITLES[b]}  [{r.name}]\n{'=' * 78}")
        if b in MODELS[r.key].get("not_applicable", set()):
            summary[b] = "not applicable to this model"
            continue
        t0 = time.time()
        try:
            rec = BENCH_FUNCS[b](r)
            if rec is None:
                summary[b] = "dry run"
                continue
            rec["duration_min"] = round((time.time() - t0) / 60, 1)
            r.res_dir.mkdir(parents=True, exist_ok=True)
            with open(r.res_dir / f"{b}.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
            summary[b] = "ok"
            write_report()
            print(f"[{b}] {json.dumps(rec['metrics'], ensure_ascii=False)}")
        except Exception as e:   # keep going with the other benchmarks
            summary[b] = f"FAILED: {e}"
            print(f"\033[1;31m[{b}] FAILED: {e}\033[0m", flush=True)
    print(f"\n{'=' * 78}\nDone in {(time.time() - t_all) / 60:.1f} min  run={r.name}")
    for b, s in summary.items():
        print(f"  {TITLES[b]:22s}: {s}")
    if any(s.startswith("FAILED") for s in summary.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
