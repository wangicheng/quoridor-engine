#include "board.h"
#include <queue>
#include <cstring>
#include <algorithm>
#include <functional>

static constexpr int WALL_BITS = WALL_DIM * WALL_DIM;

inline int wall_idx(int r, int c) { return r * WALL_DIM + c; }

// -------------------- Construction --------------------

QuoridorBoard::QuoridorBoard() {
    reset();
}

void QuoridorBoard::reset() {
    p1_pos = {0, 4};
    p2_pos = {8, 4};
    h_walls.reset();
    v_walls.reset();
    p1_walls = 10;
    p2_walls = 10;
    turn = 0;
    last_p1_dist = 0.0f;
    last_p2_dist = 0.0f;
}

// -------------------- Position helpers --------------------

bool QuoridorBoard::_is_valid_pos(int r, int c) const {
    return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
}

bool QuoridorBoard::_can_move(int r1, int c1, int r2, int c2) const {
    if (!_is_valid_pos(r2, c2)) return false;

    // Vertical move (same column)
    if (c1 == c2) {
        int r_min = std::min(r1, r2);
        // h_walls[r_min][c1] blocks column c1
        if (c1 < WALL_DIM && h_walls.test(wall_idx(r_min, c1))) return false;
        // h_walls[r_min][c1-1] also blocks column c1
        if (c1 > 0 && h_walls.test(wall_idx(r_min, c1 - 1))) return false;
    }

    // Horizontal move (same row)
    if (r1 == r2) {
        int c_min = std::min(c1, c2);
        // v_walls[r1][c_min] blocks row r1
        if (r1 < WALL_DIM && v_walls.test(wall_idx(r1, c_min))) return false;
        // v_walls[r1-1][c_min] also blocks row r1
        if (r1 > 0 && v_walls.test(wall_idx(r1 - 1, c_min))) return false;
    }

    return true;
}

// -------------------- Pawn moves --------------------

int QuoridorBoard::get_legal_pawn_moves(int actions_out[4], int dest_r_out[4], int dest_c_out[4]) const {
    auto [r, c] = (turn == 0) ? p1_pos : p2_pos;
    auto [opp_r, opp_c] = (turn == 0) ? p2_pos : p1_pos;

    // Directions: 0=Up(-1,0), 1=Down(+1,0), 2=Left(0,-1), 3=Right(0,+1)
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    int count = 0;
    for (int dir = 0; dir < 4; dir++) {
        int nr = r + dr[dir];
        int nc = c + dc[dir];
        if (!_can_move(r, c, nr, nc)) continue;

        if (nr == opp_r && nc == opp_c) {
            // Jump over opponent
            int jr = nr + dr[dir];
            int jc = nc + dc[dir];
            if (_can_move(nr, nc, jr, jc)) {
                actions_out[count] = dir;
                dest_r_out[count] = jr;
                dest_c_out[count] = jc;
                count++;
            }
        } else {
            actions_out[count] = dir;
            dest_r_out[count] = nr;
            dest_c_out[count] = nc;
            count++;
        }
    }
    return count;
}

// -------------------- BFS Pathfinding --------------------

bool QuoridorBoard::has_path(int start_r, int start_c, int target_row) const {
    bool visited[BOARD_SIZE][BOARD_SIZE] = {false};
    std::queue<std::pair<int, int>> q;
    q.push({start_r, start_c});
    visited[start_r][start_c] = true;

    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};

    while (!q.empty()) {
        auto [r, c] = q.front(); q.pop();
        if (r == target_row) return true;

        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d];
            int nc = c + dc[d];
            if (_is_valid_pos(nr, nc) && !visited[nr][nc] && _can_move(r, c, nr, nc)) {
                visited[nr][nc] = true;
                q.push({nr, nc});
            }
        }
    }
    return false;
}

// -------------------- Multi-source BFS distance map --------------------

void QuoridorBoard::compute_distance_map(int target_row, float dist_out[BOARD_SIZE][BOARD_SIZE]) const {
    for (int r = 0; r < BOARD_SIZE; r++)
        for (int c = 0; c < BOARD_SIZE; c++)
            dist_out[r][c] = 999.0f;

    std::queue<std::pair<int, int>> q;
    for (int c = 0; c < BOARD_SIZE; c++) {
        dist_out[target_row][c] = 0.0f;
        q.push({target_row, c});
    }

    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};

    while (!q.empty()) {
        auto [r, c] = q.front(); q.pop();
        float d = dist_out[r][c];

        for (int dir = 0; dir < 4; dir++) {
            int nr = r + dr[dir];
            int nc = c + dc[dir];
            if (_is_valid_pos(nr, nc) && dist_out[nr][nc] == 999.0f) {
                // Note: reverse direction check: can_move from neighbor to current
                if (_can_move(nr, nc, r, c)) {
                    dist_out[nr][nc] = d + 1.0f;
                    q.push({nr, nc});
                }
            }
        }
    }
}

// -------------------- Legal Actions --------------------

std::bitset<NUM_ACTIONS> QuoridorBoard::get_legal_actions() const {
    std::bitset<NUM_ACTIONS> mask;

    // 1. Pawn moves
    int act[4], dr_[4], dc_[4];
    int n = const_cast<QuoridorBoard*>(this)->get_legal_pawn_moves(act, dr_, dc_);
    for (int i = 0; i < n; i++) {
        mask.set(act[i]);
    }

    // 2. Wall placements
    int walls_left = (turn == 0) ? p1_walls : p2_walls;
    if (walls_left > 0) {
        for (int r = 0; r < WALL_DIM; r++) {
            for (int c = 0; c < WALL_DIM; c++) {
                // H-wall
                bool h_legal = !h_walls.test(wall_idx(r, c));
                h_legal = h_legal && !(c > 0 && h_walls.test(wall_idx(r, c - 1)));
                h_legal = h_legal && !(c < WALL_DIM - 1 && h_walls.test(wall_idx(r, c + 1)));
                h_legal = h_legal && !v_walls.test(wall_idx(r, c));

                if (h_legal) {
                    const_cast<QuoridorBoard*>(this)->h_walls.set(wall_idx(r, c));
                    bool ok = has_path(p1_pos.first, p1_pos.second, 8)
                           && has_path(p2_pos.first, p2_pos.second, 0);
                    const_cast<QuoridorBoard*>(this)->h_walls.reset(wall_idx(r, c));
                    if (ok) {
                        int idx = 4 + r * WALL_DIM + c;
                        mask.set(idx);
                    }
                }

                // V-wall
                bool v_legal = !v_walls.test(wall_idx(r, c));
                v_legal = v_legal && !(r > 0 && v_walls.test(wall_idx(r - 1, c)));
                v_legal = v_legal && !(r < WALL_DIM - 1 && v_walls.test(wall_idx(r + 1, c)));
                v_legal = v_legal && !h_walls.test(wall_idx(r, c));

                if (v_legal) {
                    const_cast<QuoridorBoard*>(this)->v_walls.set(wall_idx(r, c));
                    bool ok = has_path(p1_pos.first, p1_pos.second, 8)
                           && has_path(p2_pos.first, p2_pos.second, 0);
                    const_cast<QuoridorBoard*>(this)->v_walls.reset(wall_idx(r, c));
                    if (ok) {
                        int idx = 68 + r * WALL_DIM + c;
                        mask.set(idx);
                    }
                }
            }
        }
    }

    return mask;
}

// -------------------- State Encoding --------------------

void QuoridorBoard::get_state(float state_out[STATE_CHANNELS * STATE_SPATIAL * STATE_SPATIAL],
                              std::bitset<NUM_ACTIONS>& mask_out,
                              float& p1_dist_ref, float& p2_dist_ref) {
    // Clear state tensor
    std::memset(state_out, 0, STATE_CHANNELS * STATE_SPATIAL * STATE_SPATIAL * sizeof(float));

    // Channel 0: P1 position
    {
        int idx = 0 * STATE_SPATIAL * STATE_SPATIAL + p1_pos.first * 2 * STATE_SPATIAL + p1_pos.second * 2;
        state_out[idx] = 1.0f;
    }

    // Channel 1: P2 position
    {
        int idx = 1 * STATE_SPATIAL * STATE_SPATIAL + p2_pos.first * 2 * STATE_SPATIAL + p2_pos.second * 2;
        state_out[idx] = 1.0f;
    }

    // Channel 2: H-walls (3 cells: (2r+1, 2c), (2r+1, 2c+1), (2r+1, 2c+2))
    for (int r = 0; r < WALL_DIM; r++) {
        for (int c = 0; c < WALL_DIM; c++) {
            if (h_walls.test(wall_idx(r, c))) {
                int base = 2 * STATE_SPATIAL * STATE_SPATIAL + (2 * r + 1) * STATE_SPATIAL + 2 * c;
                state_out[base] = 1.0f;
                state_out[base + 1] = 1.0f;
                state_out[base + 2] = 1.0f;
            }
        }
    }

    // Channel 3: V-walls (3 cells: (2r, 2c+1), (2r+1, 2c+1), (2r+2, 2c+1))
    for (int r = 0; r < WALL_DIM; r++) {
        for (int c = 0; c < WALL_DIM; c++) {
            if (v_walls.test(wall_idx(r, c))) {
                int base = 3 * STATE_SPATIAL * STATE_SPATIAL + (2 * r) * STATE_SPATIAL + 2 * c + 1;
                state_out[base] = 1.0f;
                state_out[base + STATE_SPATIAL] = 1.0f;
                state_out[base + 2 * STATE_SPATIAL] = 1.0f;
            }
        }
    }

    // Channel 4: P1 walls left, Channel 5: P2 walls left
    float p1w = p1_walls / 10.0f;
    float p2w = p2_walls / 10.0f;
    int ch4_base = 4 * STATE_SPATIAL * STATE_SPATIAL;
    int ch5_base = 5 * STATE_SPATIAL * STATE_SPATIAL;
    for (int i = 0; i < STATE_SPATIAL * STATE_SPATIAL; i++) {
        state_out[ch4_base + i] = p1w;
        state_out[ch5_base + i] = p2w;
    }

    // Get legal actions mask
    mask_out = get_legal_actions();

    // Channel 6: Legal action mask visualization
    for (int r = 0; r < WALL_DIM; r++) {
        for (int c = 0; c < WALL_DIM; c++) {
            int h_idx = 4 + r * WALL_DIM + c;
            int v_idx = 68 + r * WALL_DIM + c;
            if (mask_out.test(h_idx)) {
                int idx = 6 * STATE_SPATIAL * STATE_SPATIAL + (2 * r + 1) * STATE_SPATIAL + 2 * c;
                state_out[idx] = 1.0f;
            }
            if (mask_out.test(v_idx)) {
                int idx = 6 * STATE_SPATIAL * STATE_SPATIAL + (2 * r) * STATE_SPATIAL + 2 * c + 1;
                state_out[idx] = 1.0f;
            }
        }
    }

    // Distance maps (Channels 7 and 8)
    float p1_dmap[BOARD_SIZE][BOARD_SIZE];
    float p2_dmap[BOARD_SIZE][BOARD_SIZE];
    compute_distance_map(8, p1_dmap);
    compute_distance_map(0, p2_dmap);

    last_p1_dist = p1_dmap[p1_pos.first][p1_pos.second];
    last_p2_dist = p2_dmap[p2_pos.first][p2_pos.second];

    p1_dist_ref = last_p1_dist;
    p2_dist_ref = last_p2_dist;

    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            float d1 = std::min(p1_dmap[r][c], 81.0f);
            float d2 = std::min(p2_dmap[r][c], 81.0f);
            int idx7 = 7 * STATE_SPATIAL * STATE_SPATIAL + (2 * r) * STATE_SPATIAL + 2 * c;
            int idx8 = 8 * STATE_SPATIAL * STATE_SPATIAL + (2 * r) * STATE_SPATIAL + 2 * c;
            state_out[idx7] = d1 / 81.0f;
            state_out[idx8] = d2 / 81.0f;
        }
    }
}

void QuoridorBoard::get_state_tensor(float state_out[STATE_CHANNELS * STATE_SPATIAL * STATE_SPATIAL]) {
    std::bitset<NUM_ACTIONS> dummy_mask;
    float dummy_p1, dummy_p2;
    get_state(state_out, dummy_mask, dummy_p1, dummy_p2);
}

// -------------------- Step --------------------

void QuoridorBoard::step(int action_id) {
    if (action_id < 4) {
        // Pawn move
        int actions[4], dest_r[4], dest_c[4];
        int n = get_legal_pawn_moves(actions, dest_r, dest_c);
        for (int i = 0; i < n; i++) {
            if (actions[i] == action_id) {
                if (turn == 0) {
                    p1_pos = {dest_r[i], dest_c[i]};
                } else {
                    p2_pos = {dest_r[i], dest_c[i]};
                }
                break;
            }
        }
    } else if (action_id < 68) {
        // H-wall
        int r = (action_id - 4) / WALL_DIM;
        int c = (action_id - 4) % WALL_DIM;
        h_walls.set(wall_idx(r, c));
        if (turn == 0) p1_walls--; else p2_walls--;
    } else {
        // V-wall
        int r = (action_id - 68) / WALL_DIM;
        int c = (action_id - 68) % WALL_DIM;
        v_walls.set(wall_idx(r, c));
        if (turn == 0) p1_walls--; else p2_walls--;
    }

    turn = 1 - turn;
}

// -------------------- Terminal --------------------

bool QuoridorBoard::is_terminal() const {
    return p1_pos.first == 8 || p2_pos.first == 0;
}

float QuoridorBoard::get_result(int perspective) const {
    if (!is_terminal()) return 0.0f;
    bool p1_won = (p1_pos.first == 8);
    if (perspective == 0) {
        return p1_won ? 1.0f : -1.0f;
    } else {
        return p1_won ? -1.0f : 1.0f;
    }
}

// -------------------- Hash --------------------

size_t QuoridorBoard::get_hash() const {
    auto hash_combine = [](size_t seed, size_t v) -> size_t {
        return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    };

    size_t h = 0;
    h = hash_combine(h, static_cast<size_t>(p1_pos.first) << 4 | static_cast<size_t>(p1_pos.second));
    h = hash_combine(h, static_cast<size_t>(p2_pos.first) << 4 | static_cast<size_t>(p2_pos.second));

    // Pack walls into 64-bit values
    uint64_t hw = h_walls.to_ullong();
    uint64_t vw = v_walls.to_ullong();
    h = hash_combine(h, static_cast<size_t>(hw));
    h = hash_combine(h, static_cast<size_t>(vw));

    h = hash_combine(h, static_cast<size_t>(p1_walls));
    h = hash_combine(h, static_cast<size_t>(p2_walls));
    h = hash_combine(h, static_cast<size_t>(turn));

    return h;
}
