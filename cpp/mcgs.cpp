#include "mcgs.h"
#include <algorithm>
#include <numeric>
#include <cstring>

MCGS::MCGS(float c_puct, float fpu_reduction)
    : c_puct_(c_puct)
    , fpu_reduction_(fpu_reduction)
    , rng_(std::random_device{}())
{}

// Generate Dirichlet distribution samples
static std::vector<float> dirichlet_sample(float alpha, int n, std::mt19937& rng) {
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> samples(n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        samples[i] = gamma(rng);
        sum += samples[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        samples[i] *= inv_sum;
    }
    return samples;
}

void MCGS::reset_search(QuoridorBoard& root_board) {
    nodes_.clear();
    state_to_node_.clear();

    size_t root_hash = root_board.get_hash();

    // Create root node
    nodes_.emplace_back(1.0f);
    int root_id = 0;
    state_to_node_[root_hash] = root_id;
}

bool MCGS::search_step(QuoridorBoard& root_board, float* state_out, bool* mask_out, int& leaf_turn) {
    QuoridorBoard board = root_board;  // Clone for traversal
    visited_in_sim_.clear();
    current_path_.clear();

    int current = 0; // root is always 0
    size_t current_hash = board.get_hash();

    while (true) {
        current_path_.push_back(current);

        // Loop detection
        if (visited_in_sim_.count(current_hash)) {
            backup(-1.0f); // Draw
            return false;
        }
        visited_in_sim_.insert(current_hash);

        // Terminal check
        if (board.is_terminal()) {
            float val = board.get_result(board.get_turn());
            backup(-val);
            return false;
        }

        // Leaf node: Request evaluation
        if (nodes_[current].children.empty()) {
            std::bitset<NUM_ACTIONS> mask_bitset;
            board.get_state(state_out, mask_bitset, saved_p1_dist_, saved_p2_dist_);
            
            for (int i = 0; i < NUM_ACTIONS; i++) {
                mask_out[i] = mask_bitset.test(i);
                saved_mask_[i] = mask_out[i];
            }
            saved_turn_ = board.get_turn();
            leaf_turn = saved_turn_;

            return true;
        }

        // PUCT selection
        int best_action = -1;
        float best_ucb = -1e9f;
        float sqrt_total_visits = std::sqrt(static_cast<float>(nodes_[current].visit_count));

        for (auto& [action, child_id] : nodes_[current].children) {
            auto& child = nodes_[child_id];
            float q_val;
            if (child.visit_count == 0) {
                // FPU: use parent's prior_value
                q_val = nodes_[current].prior_value;
                if (action >= 4) {
                    q_val -= fpu_reduction_;  // Penalize wall moves
                }
            } else {
                q_val = -child.value();  // Negate for zero-sum
            }

            float u_val = c_puct_ * child.prior_prob * sqrt_total_visits
                          / (1.0f + static_cast<float>(child.visit_count));
            float ucb = q_val + u_val;

            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_action = action;
            }
        }

        if (best_action < 0) {
            backup(-1.0f);
            return false;
        }

        // Execute action
        board.step(best_action);
        size_t next_hash = board.get_hash();

        // Transposition table (DAG) handling
        auto it = state_to_node_.find(next_hash);
        int next_node_id;
        if (it != state_to_node_.end()) {
            next_node_id = it->second;
            nodes_[current].children[best_action] = next_node_id;
        } else {
            next_node_id = nodes_[current].children[best_action];
            state_to_node_[next_hash] = next_node_id;
        }

        current = next_node_id;
        current_hash = next_hash;
    }
}

void MCGS::expand_and_backup(float nn_value, const float* probs) {
    int leaf_id = current_path_.back();

    // Compute heuristic value
    float heuristic;
    if (saved_turn_ == 0) {
        heuristic = std::tanh((saved_p2_dist_ - saved_p1_dist_) / 3.0f);
    } else {
        heuristic = std::tanh((saved_p1_dist_ - saved_p2_dist_) / 3.0f);
    }

    // Blend: 0.8 * NN + 0.2 * heuristic
    float final_value = 0.8f * nn_value + 0.2f * heuristic;

    // Store prior value for FPU
    nodes_[leaf_id].prior_value = final_value;

    // Count valid moves
    int valid_count = 0;
    for (int i = 0; i < NUM_ACTIONS; i++) {
        if (saved_mask_[i]) valid_count++;
    }

    if (valid_count > 0) {
        // Standard AlphaZero: apply Dirichlet noise only at the root node
        bool apply_noise = (leaf_id == 0);
        std::vector<float> noise;
        if (apply_noise) {
            noise = dirichlet_sample(0.3f, valid_count, rng_);
        }

        int noise_idx = 0;
        for (int action = 0; action < NUM_ACTIONS; action++) {
            if (saved_mask_[action]) {
                float p = probs[action];
                if (apply_noise) {
                    // Mix prior with Dirichlet noise
                    p = 0.75f * p + 0.25f * noise[noise_idx++];
                }
                
                int child_id = static_cast<int>(nodes_.size());
                nodes_.emplace_back(p);
                nodes_[leaf_id].children[action] = child_id;
            }
        }
    }

    // Backpropagate leaf value
    backup(-final_value);
}

void MCGS::backup(float val) {
    float current_val = val;
    // Backpropagate up the current_path_
    for (auto it = current_path_.rbegin(); it != current_path_.rend(); ++it) {
        int node_id = *it;
        nodes_[node_id].visit_count++;
        nodes_[node_id].value_sum += current_val;
        current_val = -current_val; // Alternate sign for zero-sum turn
    }
}

std::array<float, NUM_ACTIONS> MCGS::get_action_visits() const {
    std::array<float, NUM_ACTIONS> action_visits{};
    if (!nodes_.empty()) {
        for (auto& [action, child_id] : nodes_[0].children) {
            action_visits[action] = static_cast<float>(nodes_[child_id].visit_count);
        }
    }
    return action_visits;
}
