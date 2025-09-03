import argparse, yaml, os
from datetime import datetime
from src.pipeline import run_pipeline
from src.utils import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_base = cfg["outputs"]["out_dir"]
    run_name = cfg["run"]["run_name"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["run"]["run_name"] = f"{run_name}_{ts}"
    ensure_dir(out_base)

    out_dir, lb = run_pipeline(cfg)
    print(f"Finished. Results in: {out_dir}")
    try:
        print(lb.head())
    except Exception:
        pass

if __name__ == "__main__":
    main()
