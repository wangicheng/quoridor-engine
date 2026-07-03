import numpy as np
from collections import deque

class QuoridorEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.p1_pos = (0, 4) # Top middle
        self.p2_pos = (8, 4) # Bottom middle
        self.h_walls = np.zeros((8, 8), dtype=bool)
        self.v_walls = np.zeros((8, 8), dtype=bool)
        self.p1_walls = 10
        self.p2_walls = 10
        self.turn = 0 # 0 for p1, 1 for p2
        return self.get_state()
        
    def _is_valid_pos(self, r, c):
        return 0 <= r < 9 and 0 <= c < 9
        
    def get_legal_pawn_moves(self, player):
        pos = self.p1_pos if player == 0 else self.p2_pos
        opp_pos = self.p2_pos if player == 0 else self.p1_pos
        
        r, c = pos
        moves = []
        
        # Directions: Up(0), Down(1), Left(2), Right(3)
        # Note: mapping exactly to 4 directions as per 132 action space blueprint.
        # For simplicity in this PoC, we implement 1-step orthogonal moves.
        # If moving into opponent, we jump over them if not blocked.
        
        # Check Up (-1, 0)
        if self._can_move(r, c, r-1, c):
            if (r-1, c) == opp_pos:
                if self._can_move(r-1, c, r-2, c): # Jump over
                    moves.append((0, (r-2, c)))
            else:
                moves.append((0, (r-1, c)))
                
        # Check Down (+1, 0)
        if self._can_move(r, c, r+1, c):
            if (r+1, c) == opp_pos:
                if self._can_move(r+1, c, r+2, c):
                    moves.append((1, (r+2, c)))
            else:
                moves.append((1, (r+1, c)))
                
        # Check Left (0, -1)
        if self._can_move(r, c, r, c-1):
            if (r, c-1) == opp_pos:
                if self._can_move(r, c-1, r, c-2):
                    moves.append((2, (r, c-2)))
            else:
                moves.append((2, (r, c-1)))
                
        # Check Right (0, +1)
        if self._can_move(r, c, r, c+1):
            if (r, c+1) == opp_pos:
                if self._can_move(r, c+1, r, c+2):
                    moves.append((3, (r, c+2)))
            else:
                moves.append((3, (r, c+1)))
                
        return moves

    def _can_move(self, r1, c1, r2, c2):
        if not self._is_valid_pos(r2, c2): return False
        
        # Moving vertically
        if c1 == c2:
            r_min = min(r1, r2)
            # Check h_wall at r_min, blocking between r_min and r_min+1
            # A wall at (r_min, c) or (r_min, c-1) would block this
            if c1 < 8 and self.h_walls[r_min, c1]: return False
            if c1 > 0 and self.h_walls[r_min, c1-1]: return False
            
        # Moving horizontally
        if r1 == r2:
            c_min = min(c1, c2)
            if r1 < 8 and self.v_walls[r1, c_min]: return False
            if r1 > 0 and self.v_walls[r1-1, c_min]: return False
            
        return True

    def has_path(self, start_pos, target_row):
        """BFS to check if a player can reach their target row."""
        queue = deque([start_pos])
        visited = set([start_pos])
        
        while queue:
            r, c = queue.popleft()
            if r == target_row:
                return True
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if self._is_valid_pos(nr, nc) and (nr, nc) not in visited:
                    if self._can_move(r, c, nr, nc):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def get_legal_actions(self):
        """Returns a boolean array of length 132 for legal actions."""
        # 0-3: Pawn moves, 4-67: H-walls, 68-131: V-walls
        mask = np.zeros(132, dtype=bool)
        
        # 1. Pawn moves
        pawn_moves = self.get_legal_pawn_moves(self.turn)
        for move_idx, _ in pawn_moves:
            mask[move_idx] = True
            
        # 2. Walls
        walls_left = self.p1_walls if self.turn == 0 else self.p2_walls
        if walls_left > 0:
            for r in range(8):
                for c in range(8):
                    # Check H-wall legality
                    # Cannot overlap with existing H-walls
                    # Cannot cross existing V-wall at same intersection
                    if not self.h_walls[r, c] and not (c > 0 and self.h_walls[r, c-1]) and not (c < 7 and self.h_walls[r, c+1]):
                        if not self.v_walls[r, c]:
                            # Temp place and check BFS
                            self.h_walls[r, c] = True
                            if self.has_path(self.p1_pos, 8) and self.has_path(self.p2_pos, 0):
                                idx = 4 + r * 8 + c
                                mask[idx] = True
                            self.h_walls[r, c] = False
                            
                    # Check V-wall legality
                    if not self.v_walls[r, c] and not (r > 0 and self.v_walls[r-1, c]) and not (r < 7 and self.v_walls[r+1, c]):
                        if not self.h_walls[r, c]:
                            self.v_walls[r, c] = True
                            if self.has_path(self.p1_pos, 8) and self.has_path(self.p2_pos, 0):
                                idx = 68 + r * 8 + c
                                mask[idx] = True
                            self.v_walls[r, c] = False
                            
        return mask

    def _compute_distance_map(self, target_row):
        dist = np.full((9, 9), 999, dtype=np.float32)
        queue = deque()
        for c in range(9):
            queue.append((target_row, c))
            dist[target_row, c] = 0
            
        while queue:
            r, c = queue.popleft()
            d = dist[r, c]
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if self._is_valid_pos(nr, nc) and dist[nr, nc] == 999:
                    if self._can_move(nr, nc, r, c):
                        dist[nr, nc] = d + 1
                        queue.append((nr, nc))
        return dist

    def get_state(self):
        """Build the 17x17x9 tensor for Neural Net"""
        state = np.zeros((9, 17, 17), dtype=np.float32)
        
        # 0: P1 Pos, 1: P2 Pos
        state[0, self.p1_pos[0]*2, self.p1_pos[1]*2] = 1.0
        state[1, self.p2_pos[0]*2, self.p2_pos[1]*2] = 1.0
        
        # 2: H-walls, 3: V-walls
        for r in range(8):
            for c in range(8):
                if self.h_walls[r, c]:
                    state[2, 2*r+1, 2*c:2*c+3] = 1.0
                if self.v_walls[r, c]:
                    state[3, 2*r:2*r+3, 2*c+1] = 1.0
                    
        # 4, 5: Walls left (broadcasted)
        state[4, :, :] = self.p1_walls / 10.0
        state[5, :, :] = self.p2_walls / 10.0
        
        mask = self.get_legal_actions()
        
        for r in range(8):
            for c in range(8):
                h_idx = 4 + r * 8 + c
                v_idx = 68 + r * 8 + c
                if mask[h_idx]: state[6, 2*r+1, 2*c] = 1.0
                if mask[v_idx]: state[6, 2*r, 2*c+1] = 1.0
                
        # 7: P1 distance map, 8: P2 distance map
        p1_dmap = self._compute_distance_map(8)
        p2_dmap = self._compute_distance_map(0)
        
        self.last_p1_dist = p1_dmap[self.p1_pos]
        self.last_p2_dist = p2_dmap[self.p2_pos]
        
        for r in range(9):
            for c in range(9):
                # Normalize distances slightly (max path is usually around ~81)
                # Cap at 81.0 so unreachables don't explode the network input
                d1 = min(p1_dmap[r, c], 81.0)
                d2 = min(p2_dmap[r, c], 81.0)
                state[7, 2*r, 2*c] = d1 / 81.0
                state[8, 2*r, 2*c] = d2 / 81.0
                
        return state, mask

    def clone(self):
        new_env = QuoridorEnv()
        new_env.p1_pos = self.p1_pos
        new_env.p2_pos = self.p2_pos
        new_env.h_walls = self.h_walls.copy()
        new_env.v_walls = self.v_walls.copy()
        new_env.p1_walls = self.p1_walls
        new_env.p2_walls = self.p2_walls
        new_env.turn = self.turn
        return new_env
        
    def get_hash(self):
        # A simple string or tuple hash representing the current state uniquely
        # Transposition depends on pieces positions, walls, walls left and turn
        h_bytes = self.h_walls.tobytes()
        v_bytes = self.v_walls.tobytes()
        return hash((self.p1_pos, self.p2_pos, h_bytes, v_bytes, self.p1_walls, self.p2_walls, self.turn))
        
    def is_terminal(self):
        return self.p1_pos[0] == 8 or self.p2_pos[0] == 0
        
    def get_result(self, perspective=0):
        # Returns 1 if 'perspective' won, -1 if lost, 0 otherwise
        if not self.is_terminal(): return 0
        p1_won = (self.p1_pos[0] == 8)
        if perspective == 0:
            return 1 if p1_won else -1
        else:
            return -1 if p1_won else 1

    def step(self, action_id):
        # action_id: 0-3 for Pawn, 4-67 for H-wall, 68-131 for V-wall
        if action_id < 4:
            # Pawn move
            moves = self.get_legal_pawn_moves(self.turn)
            for m_id, dest in moves:
                if m_id == action_id:
                    if self.turn == 0:
                        self.p1_pos = dest
                    else:
                        self.p2_pos = dest
                    break
        elif action_id < 68:
            # H-wall
            r = (action_id - 4) // 8
            c = (action_id - 4) % 8
            self.h_walls[r, c] = True
            if self.turn == 0: self.p1_walls -= 1
            else: self.p2_walls -= 1
        else:
            # V-wall
            r = (action_id - 68) // 8
            c = (action_id - 68) % 8
            self.v_walls[r, c] = True
            if self.turn == 0: self.p1_walls -= 1
            else: self.p2_walls -= 1
            
        self.turn = 1 - self.turn

if __name__ == "__main__":
    env = QuoridorEnv()
    state, mask = env.get_state()
    print("State shape:", state.shape)
    print("Legal actions count:", mask.sum(), "/ 132")
