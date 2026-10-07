"""Run a single-iteration ratio-gate pilot and evaluate after training finishes."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "checkpoints/corga_sparse_hr_ratio_gate_single_3k"
DIAG = ROOT / "diagnostics/sparse_hr_diagnosis"
NAME = "corga_sparse_hr_ratio_gate_single_3k"


def main():
    global OUTPUT, NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, choices=(64, 128))
    parser.add_argument("--gpu", type=int, choices=(0, 2, 3), default=0)
    parser.add_argument("--scorer-arm", choices=("plain", "context"))
    parser.add_argument("--feature-arm", choices=("frozen", "trainable"))
    args = parser.parse_args()
    capacity = args.capacity
    if sum(bool(v) for v in (capacity, args.scorer_arm, args.feature_arm)) > 1:
        raise ValueError("Choose one experiment type")
    steps = 5000 if capacity or args.scorer_arm or args.feature_arm else 3000
    if capacity:
        NAME = f"corga_sparse_hr_capacity{capacity}_single_5k"
        OUTPUT = ROOT / "checkpoints" / NAME
    if args.scorer_arm:
        NAME = f"corga_sparse_hr_scorer_{args.scorer_arm}_single_5k"
        OUTPUT = ROOT / "checkpoints" / NAME
    if args.feature_arm:
        NAME = f"corga_sparse_hr_feature_{args.feature_arm}_single_5k"
        OUTPUT = ROOT / "checkpoints" / NAME
    os.chdir(ROOT)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --gpu")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    (ROOT / "nohups").mkdir(exist_ok=True)
    status = {"state": "training", "gpu": args.gpu, "steps_target": steps,
              "batch_size": 2, "queries_per_image": 2048, "hr_iters": 1,
              "learning_rate": 1e-4, "gate_loss_weight": .1,
              "gate_target": "ratio", "trainable_parameters": "candidate_gate only",
              "selection": "minimum Edge +/-2 EPE on disjoint 100 TEST images",
              "report": "original fixed 100 TEST images, best checkpoint only"}
    if capacity:
        status.update(hidden_channels=capacity, context_channels=capacity,
                      trainable_parameters="hidden/context encoders, GRU, candidate_gate",
                      initialization="fresh structure in both arms; common frozen feature and scorer weights")
    if args.scorer_arm:
        status.update(hidden_channels=64, context_channels=64,
                      context_scoring=args.scorer_arm == "context", candidate_loss_weight=.2,
                      trainable_parameters="hidden/context encoders, GRU, gate and candidate scorer",
                      initialization="capacity64 step5000, context score correction starts at zero")
    if args.feature_arm:
        status.update(hidden_channels=64, context_channels=64, context_scoring=True,
                      train_feature=args.feature_arm == "trainable", feature_lr=1e-5,
                      candidate_loss_weight=.2, seed=43,
                      trainable_parameters="scored HR structure; shallow CNN only in trainable arm",
                      initialization="context scorer step5000; preserve all learned weights")

    def save_status():
        temporary = OUTPUT / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(OUTPUT / "status.json")

    def run(arguments, log):
        with (ROOT / "nohups" / log).open("w") as stream:
            subprocess.run([sys.executable, *arguments], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    save_status()
    try:
        initial = ("checkpoints/corga_sparse_hr_ratio_gate_single_3k/refiner_step3000.pth"
                   if capacity else "checkpoints/corga_sparse_hr_candidate_gate_pilot1k/refiner_step1000.pth")
        if args.scorer_arm:
            initial = "checkpoints/corga_sparse_hr_capacity64_single_5k/refiner_step5000.pth"
        if args.feature_arm:
            initial = "checkpoints/corga_sparse_hr_scorer_context_single_5k/refiner_step5000.pth"
        training = ["scripts/corga/train_sparse_hr_candidate_gate.py",
             "--initial-refiner", initial,
             "--steps", str(steps), "--hr-iters", "1", "--batch-size", "2",
             "--n-query", "2048", "--lr", "1e-4", "--gate-target", "ratio",
             "--gate-loss-weight", "0.1", "--save-every", "1000",
             "--output", str(OUTPUT)]
        if capacity:
            training += ["--hidden-channels", str(capacity), "--context-channels",
                         str(capacity), "--train-scope", "structure", "--fresh-structure"]
        if args.scorer_arm:
            training += ["--train-scope", "scored", "--candidate-loss-weight", "0.2"]
            if args.scorer_arm == "context":
                training += ["--context-scoring"]
        if args.feature_arm:
            training += ["--train-scope", "scored", "--candidate-loss-weight", "0.2",
                         "--context-scoring", "--seed", "43"]
            if args.feature_arm == "trainable":
                training += ["--train-feature", "--feature-lr", "1e-5"]
        run(training, NAME + ".out")
        status["state"] = "evaluating_selection"
        save_status()
        results = []
        for step in ((1000, 3000, 5000) if capacity or args.scorer_arm or args.feature_arm else (1000, 2000, 3000)):
            prefix = f"capacity{capacity}_single" if capacity else "ratio_gate_single"
            if args.scorer_arm:
                prefix = f"scorer_{args.scorer_arm}_single"
            if args.feature_arm:
                prefix = f"feature_{args.feature_arm}_single"
            path = DIAG / f"{prefix}_step{step}_disjoint100.json"
            run(["scripts/corga/eval_sparse_hr_refiner.py", "--refiner",
                 str(OUTPUT / f"refiner_step{step}.pth"), "--hr-iters", "1",
                 "--mode", "predicted_boundary", "--band-radius", "2",
                 "--samples", "100", "--disjoint-samples", "--output", str(path)],
                f"{NAME}_eval_step{step}_disjoint100.out")
            result = json.loads(path.read_text())
            results.append({"step": step, "edge2_epe": result["metrics"]["edge2"][-1]["epe"],
                            "overall_epe": result["metrics"]["overall"][-1]["epe"],
                            "result": str(path)})
        best = min(results, key=lambda item: item["edge2_epe"])
        status.update(state="evaluating_report", selection_results=results,
                      best_step=best["step"])
        save_status()
        report = DIAG / f"{prefix}_best_fixed100.json"
        run(["scripts/corga/eval_sparse_hr_refiner.py", "--refiner",
             str(OUTPUT / f"refiner_step{best['step']}.pth"), "--hr-iters", "1",
             "--mode", "predicted_boundary", "--band-radius", "2",
             "--samples", "100", "--output", str(report)], NAME + "_eval_best_fixed100.out")
        status.update(state="complete_evaluated", report_result=str(report))
        save_status()
    except Exception as exc:
        status.update(state="failed", error=str(exc))
        save_status()
        raise


if __name__ == "__main__":
    main()
