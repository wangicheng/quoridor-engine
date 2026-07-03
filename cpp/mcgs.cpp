#include "mcgs.h"
#include <algorithm>
#include <numeric>
#include <cstring>

MCGS::MCGS(EvalCallback eval_fn, float c_puct, float fpu_reduction)
    : eval_fn_(std::move(eval_fn))
    , c_puct_(c_puct)
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

void MCGS::evaluate_and_expand(int node_id, QuoridorBoard& board) {
    // Encode state
    float state[STATE_CHANNELS * STATE_SPATIAL * STATE_SPATIAL];
    std::bitset<NUM_ACTIONS> mask_bitset;
    float p1_dist, p2_dist;
    board.get_state(state, mask_bitset, p1_dist, p2_dist);

    // Convert bitset to bool array for callback
    bool mask_arr[NUM_ACTIONS];
    for (int i = 0; i < NUM_ACTIONS; i++) {
        mask_arr[i] = mask_bitset.test(i);
    }

    // Call Python callback for NN evaluation
    float probs[NUM_ACTIONS];
    float nn_value;
    eval_fn_(state, mask_arr, p1_dist, p2_dist, board.get_turn(), probs, nn_value);

    // Compute heuristic value
    float heuristic;
    if (board.get_turn() == 0) {
        heuristic = std::tanh((p2_dist - p1_dist) / 3.0f);
    } else {
        heuristic = std::tanh((p1_dist - p2_dist) / 3.0f);
    }

    // Blend: 0.8 * NN + 0.4 * heuristic
    float final_value = 0.8f * nn_value + 0.2f * heuristic;

    // Store prior value for FPU
    nodes_[node_id].prior_value = final_value;

    // Count valid moves
    int valid_count = 0;
    for (int i = 0; i < NUM_ACTIONS; i++) {
        if (mask_arr[i]) valid_count++;
    }

    if (valid_count > 0) {
        // Generate Dirichlet noise
        auto noise = dirichlet_sample(0.3f, valid_count, rng_);

        int noise_idx = 0;
        for (int action = 0; action < NUM_ACTIONS; action++) {
            if (mask_arr[action]) {
                // Mix prior with Dirichlet noise: 0.75 * prob + 0.25 * noise
                float p = 0.75f * probs[action] + 0.25f * noise[noise_idx];
                int child_id = static_cast<int>(nodes_.size());
                nodes_.emplace_back(p);
                nodes_[node_id].children[action] = child_id;
                noise_idx++;
            }
        }
    }

    // Backpropagate leaf value
    nodes_[node_id].visit_count++;
    nodes_[node_id].value_sum += final_value;
}

float MCGS::simulate(int node_id, QuoridorBoard& board,
                     std::unordered_set<size_t>& visited) {
    size_t current_hash = board.get_hash();

    // Loop detection
    if (visited.count(current_hash)) {
        return -1.0f;
    }
    visited.insert(current_hash);

    // Terminal check
    if (board.is_terminal()) {
        float val = board.get_result(board.get_turn());
        nodes_[node_id].visit_count++;
        nodes_[node_id].value_sum += val;
        return -val;
    }

    // Leaf node: evaluate and expand
    if (nodes_[node_id].children.empty()) {
        evaluate_and_expand(node_id, board);
        return -(nodes_[node_id].value_sum / nodes_[node_id].visit_count);
    }

    // PUCT selection
    int best_action = -1;
    float best_ucb = -1e9f;
    float sqrt_total_visits = std::sqrt(static_cast<float>(nodes_[node_id].visit_count));

    for (auto& [action, child_id] : nodes_[node_id].children) {
        auto& child = nodes_[child_id];
        float q_val;
        if (child.visit_count == 0) {
            // FPU: use parent's prior_value
            q_val = nodes_[node_id].prior_value;
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
        // No legal action (shouldn't happen in normal play)
        return -1.0f;
    }

    // Execute action
    board.step(best_action);
    size_t next_hash = board.get_hash();

    // Transposition table (DAG) handling
    auto it = state_to_node_.find(next_hash);
    if (it != state_to_node_.end()) {
        nodes_[node_id].children[best_action] = it->second;
    } else {
        state_to_node_[next_hash] = nodes_[node_id].children[best_action];
    }

    int next_node_id = nodes_[node_id].children[best_action];
    float val = simulate(next_node_id, board, visited);

    nodes_[node_id].visit_count++;
    nodes_[node_id].value_sum += val;

    return -val;
}

std::array<float, NUM_ACTIONS> MCGS::search(QuoridorBoard& board, int num_simulations) {
    // Reset search state
    nodes_.clear();
    state_to_node_.clear();

    size_t root_hash = board.get_hash();

    // Create root node
    nodes_.emplace_back(1.0f);
    int root_id = 0;
    state_to_node_[root_hash] = root_id;

    // Evaluate root
    evaluate_and_expand(root_id, board);

    // Run simulations
    for (int sim = 0; sim < num_simulations; sim++) {
        QuoridorBoard cloned_board = board;  // Copy
        std::unordered_set<size_t> visited;
        simulate(root_id, cloned_board, visited);
    }

    // Collect visit counts
    std::array<float, NUM_ACTIONS> action_visits{};
    for (auto& [action, child_id] : nodes_[root_id].children) {
        action_visits[action] = static_cast<float>(nodes_[child_id].visit_count);
    }

    return action_visits;
}
