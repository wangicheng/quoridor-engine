#pragma once

#include <array>
#include <bitset>
#include <cstring>
#include <queue>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <string>

static constexpr int BOARD_SIZE = 9;
static constexpr int WALL_DIM = 8;
static constexpr int STATE_CHANNELS = 9;
static constexpr int STATE_SPATIAL = 17;
static constexpr int NUM_ACTIONS = 132;

class QuoridorBoard {
public:
    QuoridorBoard();
    QuoridorBoard(const QuoridorBoard& other) = default;
    QuoridorBoard& operator=(const QuoridorBoard& other) = default;

    void reset();

    // Core game logic
    bool _is_valid_pos(int r, int c) const;
    bool _can_move(int r1, int c1, int r2, int c2) const;

    // Pawn moves: returns (action_id, dest_r, dest_c) triples
    // Action IDs: 0=up, 1=down, 2=left, 3=right
    int get_legal_pawn_moves(int actions_out[4], int dest_r_out[4], int dest_c_out[4]) const;

    // BFS from start_pos; returns true if target_row is reachable
    bool has_path(int start_r, int start_c, int target_row) const;

    // Returns boolean mask (size 132)
    std::bitset<NUM_ACTIONS> get_legal_actions() const;

    // Compute distance map from target_row to all cells (multi-source BFS)
    // dist_out must be 9x9 float array
    void compute_distance_map(int target_row, float dist_out[BOARD_SIZE][BOARD_SIZE]) const;

    // Encode full state for NN (9 x 17 x 17). Returns state + mask.
    // Also outputs p1_dist/p2_dist for heuristic computation.
    // Caller provides pre-allocated buffers.
    void get_state(float state_out[STATE_CHANNELS * STATE_SPATIAL * STATE_SPATIAL],
                   std::bitset<NUM_ACTIONS>& mask_out,
                   float& p1_dist_out, float& p2_dist_out);

    // Fast: get state tensor only (no mask recomputation if cached)
    void get_state_tensor(float state_out[STATE_CHANNELS * STATE_SPATIAL * STATE_SPATIAL]);

    // Execute action
    void step(int action_id);

    // Terminal check
    bool is_terminal() const;
    float get_result(int perspective) const;

    // Hash for transposition table
    size_t get_hash() const;

    // Accessors
    int get_turn() const { return turn; }
    std::pair<int, int> get_p1_pos() const { return p1_pos; }
    std::pair<int, int> get_p2_pos() const { return p2_pos; }
    int get_p1_walls() const { return p1_walls; }
    int get_p2_walls() const { return p2_walls; }
    float get_last_p1_dist() const { return last_p1_dist; }
    float get_last_p2_dist() const { return last_p2_dist; }

    const std::bitset<WALL_DIM * WALL_DIM>& get_h_walls() const { return h_walls; }
    const std::bitset<WALL_DIM * WALL_DIM>& get_v_walls() const { return v_walls; }

private:
    std::pair<int, int> p1_pos;
    std::pair<int, int> p2_pos;
    std::bitset<WALL_DIM * WALL_DIM> h_walls;  // index = r * 8 + c
    std::bitset<WALL_DIM * WALL_DIM> v_walls;
    int p1_walls;
    int p2_walls;
    int turn;  // 0 = P1, 1 = P2

    // Cached distance values from last get_state() call
    float last_p1_dist;
    float last_p2_dist;
};
