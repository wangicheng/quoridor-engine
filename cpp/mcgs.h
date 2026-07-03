#pragma once

#include "board.h"
#include <functional>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cmath>
#include <array>

struct MCGSNode {
    float prior_prob;
    int visit_count = 0;
    float value_sum = 0.0f;
    float prior_value = 0.0f;  // NN value from parent's eval (for FPU)
    std::unordered_map<int, int> children;  // action_id -> child_node_id

    MCGSNode() : prior_prob(1.0f) {}
    explicit MCGSNode(float prob) : prior_prob(prob) {}

    float value() const {
        if (visit_count == 0) return prior_value;
        return value_sum / static_cast<float>(visit_count);
    }
};

// Callback signature: (state_9x17x17_array, mask_132_array) -> (probs_132_array, value_scalar)
using EvalCallback = std::function<void(
    const float* state,        // 9*17*17 floats
    const bool* mask,          // 132 bools
    float p1_dist,             // heuristic distance for P1
    float p2_dist,             // heuristic distance for P2
    int turn,                  // whose turn it is (0=P1, 1=P2)
    float* probs_out,          // 132 floats output
    float& value_out           // scalar value output
)>;

class MCGS {
public:
    MCGS(EvalCallback eval_fn, float c_puct = 1.5f, float fpu_reduction = 0.01f);

    // Run search from given board state. Returns visit counts per action.
    std::array<float, NUM_ACTIONS> search(QuoridorBoard& board, int num_simulations);

private:
    float simulate(int node_id, QuoridorBoard& board,
                   std::unordered_set<size_t>& visited);

    void evaluate_and_expand(int node_id, QuoridorBoard& board);

    EvalCallback eval_fn_;
    float c_puct_;
    float fpu_reduction_;

    // Transposition table: state_hash -> node_id
    std::unordered_map<size_t, int> state_to_node_;

    // All nodes stored contiguously (stable indices)
    std::vector<MCGSNode> nodes_;

    // RNG for Dirichlet noise
    std::mt19937 rng_;
};
