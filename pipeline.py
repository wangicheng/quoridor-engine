import os
import shutil
import subprocess
import time
import glob
import re

def run_self_play(games=200, iteration=1):
    print(f"\n--- Starting Self-Play Phase ({games} games, iteration {iteration}) ---")
    subprocess.run(
        ["uv", "run", "self_play.py", "--games", str(games), "--iteration", str(iteration)],
        check=True
    )

def run_training():
    print("\n--- Starting Training Phase ---")
    subprocess.run(["uv", "run", "train.py"], cwd="nn", check=True)

def save_checkpoint(iteration):
    src = "models/best_model.pth"
    dst = f"models/checkpoint_{iteration}.pth"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Checkpoint saved: {dst}")

def get_start_iteration():
    os.makedirs("models", exist_ok=True)
    checkpoint_files = glob.glob("models/checkpoint_*.pth")
    max_iter = 0
    for f in checkpoint_files:
        match = re.search(r'checkpoint_(\d+)\.pth', os.path.basename(f))
        if match:
            num = int(match.group(1))
            if num > max_iter:
                max_iter = num
    return max_iter + 1

def main():
    print("========================================")
    print(" Quoridor RL Training Pipeline Started ")
    print("========================================")

    num_iterations = 10
    games_per_iter = 1000

    os.makedirs("models", exist_ok=True)
    
    start_iter = get_start_iteration()
    print(f"Resuming from Iteration {start_iter}")

    for iteration in range(start_iter, num_iterations + 1):
        print(f"\n>>> RL Iteration {iteration}/{num_iterations} <<<")

        # 1. Data Generation with mixed opponents
        run_self_play(games=games_per_iter, iteration=iteration)

        # 2. Model Update
        run_training()

        # 3. Archive the trained model as a historical checkpoint
        save_checkpoint(iteration)

    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()
