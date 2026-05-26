import os
import sys
import json
import time
import subprocess
import datetime
from pathlib import Path

# ===================================================================
# CONFIGURATION
# ===================================================================

ABLATIONS = [
    "baseline",
    "clip_only",
    "no_cross_attention",
    "mean_pooling",
    "no_qformer",
    "no_memory"
]

GPUS = [0, 1, 2]
WORK_DIR = "/home/drive2/user1_workspace/imageweave_project"
LOGS_DIR = os.path.join(WORK_DIR, "logs")
REPORT_PATH = os.path.join(WORK_DIR, "artifacts", "audit_report.md")

# Ensure artifacts directory exists
os.makedirs(os.path.join(WORK_DIR, "artifacts"), exist_ok=True)

# State tracking
running_processes = {}  # gpu_id -> {"ablation": name, "process": Popen_object}
completed_ablations = set()
started_ablations = set()

# Initialize from currently running tmux sessions (if any)
# We know 0, 1, 2 are currently running no_qformer, no_cross_attention, no_memory via tmux
# But to make it uniform, we will track them by checking if their tmux session exists.

def get_running_tmux_sessions():
    try:
        output = subprocess.check_output(["tmux", "ls"], text=True, stderr=subprocess.DEVNULL)
        return [line.split(":")[0] for line in output.strip().split("\n") if line]
    except subprocess.CalledProcessError:
        return []

def launch_ablation(ablation, gpu_id):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Launching {ablation} on GPU {gpu_id}")
    # Launch via run_train.sh
    cmd = f"bash run_train.sh {ablation} {gpu_id}"
    subprocess.run(cmd, shell=True, cwd=WORK_DIR)
    started_ablations.add(ablation)

def generate_audit_report():
    report_lines = []
    report_lines.append("# ImageWeave Ablation Audit Report")
    report_lines.append(f"*Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    report_lines.append("\nThis report continuously monitors all ablations to identify overfitting and performance bottlenecks.")
    
    summary_metrics = {}

    for ablation in ABLATIONS:
        log_path = os.path.join(LOGS_DIR, f"training_log_{ablation}.json")
        
        status = "Waiting"
        if ablation in completed_ablations:
            status = "Completed"
        elif ablation in started_ablations or f"imageweave-{ablation}" in get_running_tmux_sessions():
            status = "Running"

        report_lines.append(f"\n## Ablation: `{ablation}` ({status})")
        
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    history = json.load(f)
                
                if not history:
                    report_lines.append("> Waiting for first epoch to complete...")
                    continue
                
                report_lines.append("| Epoch | Train R@1 | Val R@1 | Gap (Overfit) | Avg MRR | Health |")
                report_lines.append("|-------|-----------|---------|---------------|---------|--------|")
                
                best_mrr = 0.0
                for ep in history:
                    ep_num = ep.get("epoch", 0)
                    t_r1 = ep.get("train_r1", 0.0) * 100
                    v_r1 = ep.get("val_i2t_r1", 0.0) * 100
                    gap = t_r1 - v_r1
                    mrr = ep.get("val_avg_mrr", 0.0)
                    
                    best_mrr = max(best_mrr, mrr)
                    
                    health = "🟢 Healthy"
                    if gap > 50:
                        health = "🔴 Severe Overfit"
                    elif gap > 20:
                        health = "🟡 Overfitting"
                    elif mrr < 0.01 and ep_num > 3:
                        health = "🟠 Stalled"
                        
                    report_lines.append(f"| {ep_num:02d} | {t_r1:5.2f}% | {v_r1:5.2f}% | {gap:5.2f}% | {mrr:.4f} | {health} |")
                
                summary_metrics[ablation] = {
                    "best_mrr": best_mrr,
                    "last_gap": gap,
                    "epochs": history[-1].get("epoch", 0)
                }
            except Exception as e:
                report_lines.append(f"> Error reading logs: {e}")
        else:
            report_lines.append("> No logs generated yet.")

    # Generate Final Recommendation if all done
    if len(completed_ablations) == len(ABLATIONS):
        report_lines.append("\n## 🏁 Final Architectural Recommendation")
        report_lines.append("All ablations have completed. Based on the audit above:")
        report_lines.append("\n| Ablation | Best MRR | Final Gap | Verdict |")
        report_lines.append("|----------|----------|-----------|---------|")
        for ab, metrics in summary_metrics.items():
            report_lines.append(f"| `{ab}` | {metrics['best_mrr']:.4f} | {metrics['last_gap']:.2f}% | |")
        report_lines.append("\n**Next Step**: The user should review this data to construct the final optimal training configuration.")

    # Write atomic
    temp_path = REPORT_PATH + ".tmp"
    with open(temp_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    os.replace(temp_path, REPORT_PATH)


def main():
    print("Starting automated ablation auditor...")
    
    # Mark initially running sessions
    initial_sessions = get_running_tmux_sessions()
    for ab in ABLATIONS:
        if f"imageweave-{ab}" in initial_sessions:
            started_ablations.add(ab)
            print(f"Discovered already running ablation: {ab}")
            
    while True:
        current_sessions = get_running_tmux_sessions()
        
        # Check for completed sessions
        for ab in list(started_ablations):
            if ab not in completed_ablations and f"imageweave-{ab}" not in current_sessions:
                # Give it a second check just to be sure it didn't just flap
                time.sleep(2)
                if f"imageweave-{ab}" not in get_running_tmux_sessions():
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Ablation completed: {ab}")
                    completed_ablations.add(ab)
        
        # Launch new ones if GPUs are available
        running_count = sum(1 for ab in started_ablations if ab not in completed_ablations)
        
        # We assume initial runs are on GPUs 0, 1, 2. For new runs, we just assign sequentially.
        # This is a naive scheduler: it just launches up to len(GPUS) concurrent jobs.
        for ab in ABLATIONS:
            if ab not in started_ablations and running_count < len(GPUS):
                # Find an available GPU (simplistic approach: just use running_count as index, 
                # or better, check which GPU isn't used by checking running ablations.
                # Actually, since tmux hides the process, let's just use `nvidia-smi` to find free GPU 
                # or just track locally. For simplicity, we just assign GPU `running_count` index.
                # A robust way:
                gpu_id = running_count % len(GPUS)
                launch_ablation(ab, gpu_id)
                running_count += 1
                time.sleep(10)  # Prevent race condition on train_config.py

        # Update the markdown report
        generate_audit_report()
        
        if len(completed_ablations) == len(ABLATIONS):
            print("All ablations completed! Final report generated.")
            break
            
        time.sleep(60)

if __name__ == "__main__":
    main()
