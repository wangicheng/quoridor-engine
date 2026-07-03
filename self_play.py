import torch
import numpy as np
import time
import os
import glob
import re
import argparse
from quoridor_core import QuoridorBoard, MCGS
from nn.model import QuoridorNet

def get_board_string(board):
    out = []
    out.append(f"Turn: {'Player 1' if board.turn == 0 else 'Player 2'}")
    out.append(f"P1 Walls: {board.p1_walls}, P2 Walls: {board.p2_walls}")

    p1_r, p1_c = board.p1_pos
    p2_r, p2_c = board.p2_pos

    for r in range(17):
        row_str = ""
        for c in range(17):
            if r % 2 == 0 and c % 2 == 0:
                pr, pc = r // 2, c // 2
                if (pr, pc) == (p1_r, p1_c): row_str += "1"
                elif (pr, pc) == (p2_r, p2_c): row_str += "2"
                else: row_str += "."
            elif r % 2 == 0 and c % 2 != 0:
                pr, pc = r // 2, c // 2
                if pr < 8 and pc < 8 and board.has_v_wall(pr, pc): row_str += "|"
                elif pr > 0 and pc < 8 and board.has_v_wall(pr-1, pc): row_str += "|"
                else: row_str += " "
            elif r % 2 != 0 and c % 2 == 0:
                pr, pc = r // 2, c // 2
                if pr < 8 and pc < 8 and board.has_h_wall(pr, pc): row_str += "-"
                elif pr < 8 and pc > 0 and board.has_h_wall(pr, pc-1): row_str += "-"
                else: row_str += " "
            else:
                pr, pc = r // 2, c // 2
                if pr < 8 and pc < 8 and (board.has_h_wall(pr, pc) or board.has_v_wall(pr, pc)): row_str += "+"
                elif pr < 8 and pc > 0 and board.has_h_wall(pr, pc-1): row_str += "+"
                elif pr > 0 and pc < 8 and board.has_v_wall(pr-1, pc): row_str += "+"
                else: row_str += "+"
        out.append(row_str)
    out.append("=" * 20)
    return "\n".join(out)

def generate_bottleneck_target(p1_path, p2_path, winner):
    target = np.zeros((1, 17, 17), dtype=np.float32)
    path = p1_path if winner == 1 else p2_path
    for r, c in path:
        target[0, r*2, c*2] = 1.0
    return target

def make_evaluate_fn(model_p1, model_p2, device):
    def evaluate_fn(state, mask, p1_dist, p2_dist, turn):
        model = model_p1 if turn == 0 else model_p2
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, nn_value, _, _ = model(state_tensor, mask_tensor)
            probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            nn_value = nn_value.item()
        return probs, nn_value
    return evaluate_fn

def play_game(model_p1, model_p2, device, num_simulations=20, max_steps=200, render=False):
    board = QuoridorBoard()

    trajectory = []
    p1_path = [board.p1_pos]
    p2_path = [board.p2_pos]

    step_count = 0
    transcript = []

    transcript.append("=== Initial State ===")
    transcript.append(get_board_string(board))

    eval_fn = make_evaluate_fn(model_p1, model_p2, device)

    while not board.is_terminal() and step_count < max_steps:
        state, mask, _, _ = board.get_state()
        turn = board.turn

        mcgs = MCGS(eval_fn, c_puct=1.5, fpu_reduction=0.01)
        action_visits = mcgs.search(board, num_simulations=num_simulations)

        # Calculate target policy (distribution of visits)
        total_visits = np.sum(action_visits)
        target_policy = action_visits / total_visits if total_visits > 0 else action_visits

        # Increase temperature phase to 30 steps to ensure walls and diverse moves are played
        if step_count < 30:
            probs = target_policy
            best_action = int(np.random.choice(132, p=probs))
        else:
            best_action = int(np.argmax(action_visits))

        # Determine action name for transcript
        if best_action < 4:
            action_name = "Pawn Move"
        elif best_action < 68:
            action_name = f"H-Wall at {best_action-4}"
        else:
            action_name = f"V-Wall at {best_action-68}"

        transcript.append(f"\nStep {step_count+1}: Player {turn+1} chose Action {best_action} ({action_name})")

        # Save step data temporarily
        trajectory.append({
            'state': state,
            'mask': mask,
            'target_policy': target_policy,
            'turn': turn,
            'action': best_action
        })

        board.step(best_action)

        if board.turn == 1: # P1 just moved (turn swapped)
            p1_path.append(board.p1_pos)
        else:
            p2_path.append(board.p2_pos)

        board_str = get_board_string(board)
        transcript.append(board_str)
        if render:
            print(f"\nStep {step_count+1}: Player {turn+1} chose Action {best_action} ({action_name})")
            print(board_str)

        step_count += 1

    # Game Over, Retroactive Target Calculation
    if board.is_terminal():
        winner = 1 if board.p1_pos[0] == 8 else 2
        transcript.append(f"\nGame Over! Winner: Player {winner}")
    else:
        winner = 0 # Draw
        transcript.append("\nGame Over! Draw (Step Limit Reached)")

    bottleneck_target = generate_bottleneck_target(p1_path, p2_path, winner)

    # Process trajectory into tensors
    game_data = []
    for t_step, step_data in enumerate(trajectory):
        val_target = 0.0
        if winner == 1: val_target = 1.0 if step_data['turn'] == 0 else -1.0
        elif winner == 2: val_target = -1.0 if step_data['turn'] == 0 else 1.0

        dist_target = (step_count - t_step) // 2

        game_data.append({
            'state': torch.FloatTensor(step_data['state']),
            'mask': torch.BoolTensor(step_data['mask']),
            'target_policy': torch.FloatTensor(step_data['target_policy']),
            'target_value': torch.FloatTensor([val_target]),
            'target_distance': torch.FloatTensor([dist_target]),
            'target_bottleneck': torch.FloatTensor(bottleneck_target)
        })

    return game_data, winner, step_count, "\n".join(transcript)

def build_model_pool(device, iteration=None):
    os.makedirs("models", exist_ok=True)

    latest = QuoridorNet().to(device)
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        latest.load_state_dict(torch.load(model_path))
    latest.eval()

    checkpoint_files = glob.glob("models/checkpoint_*.pth")
    checkpoints = {}
    for f in checkpoint_files:
        match = re.search(r'checkpoint_(\d+)\.pth', f)
        if match:
            num = int(match.group(1))
            checkpoints[num] = f

    nums = sorted(checkpoints.keys())

    # Exclude the most recent checkpoint if it matches latest (same training result)
    if nums:
        nums = nums[:-1]

    window_nums = nums[-5:] if nums else []
    anchor_nums = nums[:-5] if len(nums) > 5 else []

    window_models = []
    for num in window_nums:
        m = QuoridorNet().to(device)
        m.load_state_dict(torch.load(checkpoints[num]))
        m.eval()
        window_models.append(m)

    anchor_models = []
    for num in anchor_nums:
        m = QuoridorNet().to(device)
        m.load_state_dict(torch.load(checkpoints[num]))
        m.eval()
        anchor_models.append(m)

    print(f"Model pool: latest (best_model.pth) + {len(window_models)} window + {len(anchor_models)} anchor checkpoints")
    return latest, window_models, anchor_models


def select_matchup(latest, window_models, anchor_models, rng):
    r = rng.random()
    if r < 0.80 or (not window_models and not anchor_models):
        return latest, latest

    if r < 0.95 and window_models:
        opponent = window_models[rng.randint(len(window_models))]
    elif anchor_models:
        opponent = anchor_models[rng.randint(len(anchor_models))]
    else:
        return latest, latest

    if rng.random() < 0.5:
        return latest, opponent
    else:
        return opponent, latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Print board to console during play")
    parser.add_argument("--games", type=int, default=1, help="Number of self-play games to generate")
    parser.add_argument("--iteration", type=int, default=None, help="Current training iteration (for checkpoint pool management)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    latest, window_models, anchor_models = build_model_pool(device, iteration=args.iteration)
    rng = np.random.RandomState()

    iteration_num = args.iteration if args.iteration is not None else 1
    data_dir = f"data/iter_{iteration_num:03d}"
    os.makedirs(data_dir, exist_ok=True)

    existing_files = glob.glob(os.path.join(data_dir, "game_*.pt"))
    existing_games = len(existing_files)
    remaining_games = args.games - existing_games

    if remaining_games <= 0:
        print(f"Self-play already completed for iteration {iteration_num} ({existing_games}/{args.games} games). Skipping.")
        return

    print(f"Resuming self-play: {existing_games} games exist. Generating {remaining_games} more games.")

    total_games = 0
    total_draws = 0

    for game_idx in range(remaining_games):
        model_p1, model_p2 = select_matchup(latest, window_models, anchor_models, rng)

        game_data, winner, steps, transcript = play_game(
            model_p1, model_p2, device,
            num_simulations=20, max_steps=100, render=args.render
        )

        timestamp = int(time.time() * 1000)
        suffix = f"{timestamp}_{game_idx}"

        if winner != 0:
            data_path = os.path.join(data_dir, f"game_{suffix}.pt")
            torch.save(game_data, data_path)
        else:
            total_draws += 1

        log_path = os.path.join(data_dir, f"game_{suffix}.txt")
        with open(log_path, "w") as f:
            f.write(transcript)

        total_games += 1
        if (game_idx + 1) % 20 == 0 or (game_idx + 1) == remaining_games:
            print(f"  {game_idx+1}/{remaining_games} games completed (draws: {total_draws})")

    print(f"Self-play finished: {total_games} new games, {total_draws} draws")

if __name__ == "__main__":
    main()
