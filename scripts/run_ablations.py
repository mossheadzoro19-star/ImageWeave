import os
import re
import subprocess
import json
import csv

CONFIG_PATH = "configs/train_config.py"
RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "master_tracking.csv")

ABLATIONS = {
    "clip_only": {
        "use_qformer": False,
        "use_cross_attention": False,
        "use_memory": False,
        "use_attention_pooling": False,
    },
    "no_cross_attention": {
        "use_qformer": True,
        "use_cross_attention": False,
        "use_memory": True,
        "use_attention_pooling": True,
    },
    "mean_pooling": {
        "use_qformer": True,
        "use_cross_attention": True,
        "use_memory": True,
        "use_attention_pooling": False,
    }
}

def update_config(ablation_name, flags):
    with open(CONFIG_PATH, "r") as f:
        content = f.read()

    # Update ablation name
    content = re.sub(r'"ablation_name": ".*?"', f'"ablation_name": "{ablation_name}"', content)
    
    # Update boolean flags
    for flag, val in flags.items():
        # Match `"flag": True,` or `"flag": False,` with potential spaces
        content = re.sub(rf'"{flag}":\s*(True|False),', f'"{flag}": {val},', content)

    with open(CONFIG_PATH, "w") as f:
        f.write(content)
    print(f"Updated config for {ablation_name}")

def log_to_csv(ablation_name, metrics):
    file_exists = os.path.isfile(CSV_PATH)
    headers = ["ablation_name", "avg_MRR", "avg_R@1", "avg_R@5", "avg_R@10", "i2t_R@1", "t2i_R@1"]
    
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        
        row = {"ablation_name": ablation_name}
        if metrics:
            row.update({
                "avg_MRR": metrics.get("Average_MRR", ""),
                "avg_R@1": metrics.get("Average_R@1", ""),
                "avg_R@5": metrics.get("Average_R@5", ""),
                "avg_R@10": metrics.get("Average_R@10", ""),
                "i2t_R@1": metrics.get("Image_to_Text", {}).get("R@1", ""),
                "t2i_R@1": metrics.get("Text_to_Image", {}).get("R@1", "")
            })
        writer.writerow(row)

def main():
    # Make sure we are in the project root by checking if configs exists
    if not os.path.exists("configs"):
        print("Please run this script from the project root: python scripts/run_ablations.py")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # The baseline is already running or complete. We just need to orchestrate the rest.
    for name, flags in ABLATIONS.items():
        print(f"\n{'='*50}\nSTARTING ABLATION: {name}\n{'='*50}\n")
        
        # 1. Update config
        update_config(name, flags)
        
        # 2. Run training
        print(f"Running training for {name}...")
        train_cmd = ["python", "train.py"]
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed for {name}: {e}")
            continue
        
        # 3. Run evaluation
        print(f"Running evaluation for {name}...")
        checkpoint_path = f"checkpoints/{name}/best_model.pth"
        output_json = f"results/{name}.json"
        eval_cmd = ["python", "evaluate.py", "--checkpoint", checkpoint_path, "--output", output_json]
        
        try:
            subprocess.run(eval_cmd, check=True)
            
            # Read metrics and log to CSV
            with open(output_json, "r") as f:
                metrics = json.load(f)
            log_to_csv(name, metrics)
            print(f"Successfully evaluated and logged {name}.")
        except Exception as e:
            print(f"Evaluation failed for {name}: {e}")
            log_to_csv(name, {})

if __name__ == "__main__":
    main()
