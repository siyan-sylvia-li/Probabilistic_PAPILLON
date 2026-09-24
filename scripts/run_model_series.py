"""Optimize and evaluate PAPILLON for every local model listed in a config CSV (paper Table 2).

For each row, starts a vLLM (or SGLang) server, runs scripts/optimize_papillon.py
on the optimization data, then evaluates the zero-shot and optimized pipelines
with scripts/evaluate_papillon.py, and shuts the server down.

CSV columns
-----------
Required:
  model              Hugging Face model ID or local path
Optional (defaults shown):
  data_file          optimization data            (default: data/PUPA_SD.csv)
  eval_file          evaluation data              (default: data/PUPA_TNB.csv)
  backend            vllm | sglang                (default: vllm)
  port               integer                      (default: 10002)
  openai_model       remote model                 (default: gpt-4o-mini)
  server_timeout     seconds                      (default: 600)
  extra_server_args  flags forwarded to the server, e.g. "--gpu-memory-utilization 0.9"

Usage:
    python scripts/run_model_series.py --config configs/models.csv
"""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from prob_papillon import DATA_DIR, REPO_ROOT

SCRIPTS_DIR = Path(__file__).resolve().parent

REQUIRED_COLS = {"model"}
DEFAULTS = {
    "data_file": str(DATA_DIR / "PUPA_SD.csv"),
    "eval_file": str(DATA_DIR / "PUPA_TNB.csv"),
    "backend": "vllm",
    "port": "10002",
    "openai_model": "gpt-4o-mini",
    "server_timeout": "600",
    "extra_server_args": "",
}


def start_server(backend: str, model: str, port: int, extra_args: list[str]) -> subprocess.Popen:
    if backend == "vllm":
        cmd = ["vllm", "serve", model, "--port", str(port), "--tensor-parallel-size", "1", *extra_args]
    else:
        cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--port", str(port), *extra_args]
    print(f"[server] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def wait_for_server(port: int, timeout: int, poll_interval: float = 5.0):
    """Block until the server's /v1/models endpoint responds, or raise on timeout."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    print(f"[server] Waiting for server on port {port} (timeout={timeout}s) …")
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                print("[server] Server is ready.")
                return
        except requests.ConnectionError:
            pass
        time.sleep(poll_interval)
    raise RuntimeError(f"Server on port {port} did not become ready within {timeout}s")


def stop_server(proc: subprocess.Popen):
    print("[server] Stopping server …")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def load_configs(csv_path: str) -> list[dict]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Config CSV is missing required columns: {missing}")

    configs = []
    for i, row in df.iterrows():
        cfg = {**DEFAULTS, **{k: v for k, v in row.items() if v != ""}}
        cfg["port"] = int(cfg["port"])
        cfg["server_timeout"] = int(cfg["server_timeout"])
        cfg["extra_server_args"] = shlex.split(cfg["extra_server_args"])
        if cfg["backend"] not in ("vllm", "sglang"):
            raise ValueError(f"Row {i}: backend must be 'vllm' or 'sglang', got '{cfg['backend']}'")
        configs.append(cfg)
    return configs


def run_series(configs: list[dict], output_dir: Path, extra_args: list[str], eval_only: bool, post_shutdown_delay: float = 5.0):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, cfg in enumerate(configs):
        name = cfg["model"].replace("/", "_")
        prompt_output = output_dir / f"{name}.json"
        common = ["--model_name", cfg["model"], "--port", str(cfg["port"]), "--openai_model", cfg["openai_model"], *extra_args]
        print(f"\n{'=' * 60}\nRun {i + 1}/{len(configs)}: {cfg['model']} ({cfg['backend']}, port {cfg['port']})\n{'=' * 60}")

        proc = start_server(cfg["backend"], cfg["model"], cfg["port"], cfg["extra_server_args"])
        try:
            wait_for_server(cfg["port"], timeout=cfg["server_timeout"])
            if not eval_only:
                subprocess.run([sys.executable, str(SCRIPTS_DIR / "optimize_papillon.py"), *common,
                                "--data_file", cfg["data_file"], "--prompt_output", str(prompt_output)], check=True)
            subprocess.run([sys.executable, str(SCRIPTS_DIR / "evaluate_papillon.py"), *common,
                            "--data_file", cfg["eval_file"], "--output_file", str(output_dir / f"{name}_before.json")])
            subprocess.run([sys.executable, str(SCRIPTS_DIR / "evaluate_papillon.py"), *common,
                            "--data_file", cfg["eval_file"], "--prompt_file", str(prompt_output),
                            "--output_file", str(output_dir / f"{name}_after.json")])
        except Exception as e:
            print(f"[error] Run failed for {cfg['model']}: {e}")
        finally:
            stop_server(proc)
            time.sleep(post_shutdown_delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="CSV file defining the run configurations")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs"), help="Where optimized prompts and evaluations are written")
    parser.add_argument("--eval_only", action="store_true", help="Skip optimization; reuse <output_dir>/<model>.json")
    parser.add_argument("--k_anon_estimator", choices=["direct", "branch"], default="direct")
    args = parser.parse_args()

    configs = load_configs(args.config)
    print(f"Loaded {len(configs)} run(s) from {args.config}")
    run_series(configs, Path(args.output_dir), ["--k_anon_estimator", args.k_anon_estimator], args.eval_only)
