#pragma once

#include "board.h"
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

class MCGS {
public:
    MCGS(float c_puct = 1.5f, float fpu_reduction = 0.01f);

    // Initialize or reset the search tree for a new board state
    void reset_search(QuoridorBoard& root_board);

    // Perform one step of search. 
    // Returns true if evaluation is needed (leaf node reached). Outputs are populated.
    // Returns false if simulation completed without needing eval (terminal state).
    bool search_step(QuoridorBoard& root_board, float* state_out, bool* mask_out, int& leaf_turn);

    // Provide neural network evaluation and complete the backpropagation for the current leaf
    void expand_and_backup(float nn_value, const float* probs);

    // Get the visit counts of the root node actions
    std::array<float, NUM_ACTIONS> get_action_visits() const;

private:
    void backup(float val);

    float c_puct_;
    float fpu_reduction_;

    std::unordered_map<size_t, int> state_to_node_;
    std::vector<MCGSNode> nodes_;
    std::mt19937 rng_;

    // Stateful variables for the current simulation
    std::vector<int> current_path_;
    std::unordered_set<size_t> visited_in_sim_;

    // Variables saved when evaluation is requested
    bool saved_mask_[NUM_ACTIONS];
    float saved_p1_dist_;
    float saved_p2_dist_;
    int saved_turn_;
};
