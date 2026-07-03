from core.env import QuoridorEnv
import numpy as np

def print_distances(env):
    state, mask = env.get_state()
    p1_dmap = state[7]
    p2_dmap = state[8]
    
    # Check current positions
    p1_r, p1_c = env.p1_pos
    p2_r, p2_c = env.p2_pos
    
    # Re-scale back to true distance for readability
    p1_dist = np.round(p1_dmap * 81.0).astype(int)
    p2_dist = np.round(p2_dmap * 81.0).astype(int)
    
    print("P1 Distance Map (Goal is row 8):")
    print(p1_dist[0::2, 0::2]) # Print 9x9 grid
    print(f"P1 current dist at {env.p1_pos}: {p1_dist[p1_r*2, p1_c*2]}")
    
    print("\nP2 Distance Map (Goal is row 0):")
    print(p2_dist[0::2, 0::2]) # Print 9x9 grid
    print(f"P2 current dist at {env.p2_pos}: {p2_dist[p2_r*2, p2_c*2]}")

if __name__ == "__main__":
    env = QuoridorEnv()
    print("=== Initial State ===")
    print_distances(env)
    
    print("\n" + "="*40 + "\n")
    
    # Place an H-wall right below P1: r=0, c=4 -> index 8
    # Turn is P1
    env.step(8) 
    print("=== After H-Wall at (0, 4) ===")
    print_distances(env)
    
    print("\n" + "="*40 + "\n")
    
    # Turn is P2. P2 places V-wall to the left of P1: r=0, c=3 -> index 0*8 + 3 + 68 = 71
    env.step(71)
    print("=== After V-Wall at (0, 3) ===")
    print_distances(env)
