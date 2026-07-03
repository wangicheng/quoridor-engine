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

class GameEnv:
    def __init__(self, model_p1, model_p2, num_simulations=20, max_steps=100, render=False):
        self.model_p1 = model_p1
        self.model_p2 = model_p2
        self.num_simulations = num_simulations
        self.max_steps = max_steps
        self.render = render

        self.board = QuoridorBoard()
        self.mcgs = MCGS(c_puct=1.5, fpu_reduction=0.01)
        self.mcgs.reset_search(self.board)
        
        self.trajectory = []
        self.p1_path = [self.board.p1_pos]
        self.p2_path = [self.board.p2_pos]
        self.step_count = 0
        self.simulations_done = 0
        
        self.transcript = []
        self.transcript.append("=== Initial State ===")
        self.transcript.append(get_board_string(self.board))
        
        self.is_done = False
        self.game_data = None
        self.winner = 0

    def get_eval_model(self, leaf_turn):
        return self.model_p1 if leaf_turn == 0 else self.model_p2

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

    # Batched Self-Play configuration
    batch_size = 32
    active_envs = []
    
    total_games_completed = 0
    total_draws = 0

    def start_new_game():
        model_p1, model_p2 = select_matchup(latest, window_models, anchor_models, rng)
        # Using 20 simulations per step as originally configured
        return GameEnv(model_p1, model_p2, num_simulations=20, max_steps=100, render=args.render)

    # Initialize batch
    for _ in range(min(batch_size, remaining_games)):
        active_envs.append(start_new_game())

    start_time = time.time()
    batch_step_count = 0

    while active_envs:
        if batch_step_count % 50 == 0:
            print(f"  [Progress] Active games: {len(active_envs):2d} | Batch MCTS steps: {batch_step_count:4d} | Completed: {total_games_completed}/{remaining_games}")
        batch_step_count += 1

        states = []
        masks = []
        models_to_eval = []
        env_indices = []

        # 1. Step MCTS for all active games
        for i, env in enumerate(active_envs):
            while env.simulations_done < env.num_simulations:
                needs_eval, state, mask, leaf_turn = env.mcgs.search_step(env.board)
                if needs_eval:
                    states.append(state)
                    masks.append(mask)
                    models_to_eval.append(env.get_eval_model(leaf_turn))
                    env_indices.append(i)
                    break # Suspend and wait for evaluation
                else:
                    env.simulations_done += 1

            if env.simulations_done >= env.num_simulations:
                # 2. Pick action, step board
                action_visits = env.mcgs.get_action_visits()
                total_visits = np.sum(action_visits)
                target_policy = action_visits / total_visits if total_visits > 0 else action_visits

                # Temperature phase: 30 steps
                if env.step_count < 30:
                    best_action = int(np.random.choice(132, p=target_policy))
                else:
                    best_action = int(np.argmax(action_visits))

                # Determine action name for transcript
                if best_action < 4:
                    action_name = "Pawn Move"
                elif best_action < 68:
                    action_name = f"H-Wall at {best_action-4}"
                else:
                    action_name = f"V-Wall at {best_action-68}"

                env.transcript.append(f"\nStep {env.step_count+1}: Player {env.board.turn+1} chose Action {best_action} ({action_name})")

                # Save trajectory step
                state_arr, mask_arr, _, _ = env.board.get_state()
                env.trajectory.append({
                    'state': state_arr,
                    'mask': mask_arr,
                    'target_policy': target_policy,
                    'turn': env.board.turn,
                    'action': best_action
                })

                env.board.step(best_action)
                
                if env.board.turn == 1:
                    env.p1_path.append(env.board.p1_pos)
                else:
                    env.p2_path.append(env.board.p2_pos)

                board_str = get_board_string(env.board)
                env.transcript.append(board_str)
                if env.render:
                    print(f"\nStep {env.step_count+1}: Player {env.board.turn+1} chose Action {best_action} ({action_name})")
                    print(board_str)

                env.step_count += 1
                env.simulations_done = 0
                env.mcgs.reset_search(env.board)

                if env.board.is_terminal() or env.step_count >= env.max_steps:
                    env.is_done = True
                    if env.board.is_terminal():
                        env.winner = 1 if env.board.p1_pos[0] == 8 else 2
                        env.transcript.append(f"\nGame Over! Winner: Player {env.winner}")
                    else:
                        env.winner = 0
                        env.transcript.append("\nGame Over! Draw (Step Limit Reached)")

        # 3. Batch evaluate collected leaves
        if states:
            unique_models = list(set(models_to_eval))
            for model in unique_models:
                # Group states by the evaluating model
                model_idx = [idx for idx, m in enumerate(models_to_eval) if m == model]
                if not model_idx: continue

                b_states = np.array([states[idx] for idx in model_idx])
                b_masks = np.array([masks[idx] for idx in model_idx])

                s_tensor = torch.FloatTensor(b_states).to(device)
                m_tensor = torch.BoolTensor(b_masks).to(device)

                with torch.no_grad():
                    policy_logits, nn_value, _, _ = model(s_tensor, m_tensor)
                    probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
                    nn_value = nn_value.cpu().numpy()

                for j, batch_idx in enumerate(model_idx):
                    env_i = env_indices[batch_idx]
                    active_envs[env_i].mcgs.expand_and_backup(float(nn_value[j].item()), probs[j])
                    active_envs[env_i].simulations_done += 1

        # 4. Handle completed games
        next_active_envs = []
        for env in active_envs:
            if env.is_done:
                bottleneck_target = generate_bottleneck_target(env.p1_path, env.p2_path, env.winner)
                
                game_data = []
                for t_step, step_data in enumerate(env.trajectory):
                    val_target = 0.0
                    if env.winner == 1: val_target = 1.0 if step_data['turn'] == 0 else -1.0
                    elif env.winner == 2: val_target = -1.0 if step_data['turn'] == 0 else 1.0

                    dist_target = (env.step_count - t_step) // 2

                    game_data.append({
                        'state': torch.FloatTensor(step_data['state']),
                        'mask': torch.BoolTensor(step_data['mask']),
                        'target_policy': torch.FloatTensor(step_data['target_policy']),
                        'target_value': torch.FloatTensor([val_target]),
                        'target_distance': torch.FloatTensor([dist_target]),
                        'target_bottleneck': torch.FloatTensor(bottleneck_target)
                    })

                timestamp = int(time.time() * 1000)
                suffix = f"{timestamp}_{total_games_completed}"

                if env.winner != 0:
                    data_path = os.path.join(data_dir, f"game_{suffix}.pt")
                    torch.save(game_data, data_path)
                else:
                    total_draws += 1

                log_path = os.path.join(data_dir, f"game_{suffix}.txt")
                with open(log_path, "w") as f:
                    f.write("\n".join(env.transcript))
                
                total_games_completed += 1
                if total_games_completed % 20 == 0 or total_games_completed == remaining_games:
                    elapsed = time.time() - start_time
                    games_per_sec = total_games_completed / elapsed
                    print(f"  {total_games_completed}/{remaining_games} games completed (draws: {total_draws}) - {games_per_sec:.2f} games/sec")
                
                # Start new game if needed
                if total_games_completed + len(next_active_envs) < remaining_games:
                    next_active_envs.append(start_new_game())
            else:
                next_active_envs.append(env)
        
        active_envs = next_active_envs

    print(f"Self-play finished: {total_games_completed} games completed, {total_draws} draws")

if __name__ == "__main__":
    main()
