#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "board.h"
#include "mcgs.h"

namespace py = pybind11;

// Helper: wrap EvalCallback from a Python function
static EvalCallback make_eval_callback(py::function py_fn) {
    return [py_fn](const float* state, const bool* mask,
                   float p1_dist, float p2_dist, int turn,
                   float* probs_out, float& value_out) {
        // Copy data into owned numpy arrays (safe for Python side)
        py::array_t<float> state_arr({STATE_CHANNELS, STATE_SPATIAL, STATE_SPATIAL});
        auto s_buf = state_arr.mutable_unchecked<3>();
        int idx = 0;
        for (int ch = 0; ch < STATE_CHANNELS; ch++) {
            for (int r = 0; r < STATE_SPATIAL; r++) {
                for (int c = 0; c < STATE_SPATIAL; c++) {
                    s_buf(ch, r, c) = state[idx++];
                }
            }
        }

        py::array_t<bool> mask_arr(NUM_ACTIONS);
        auto m_buf = mask_arr.mutable_unchecked<1>();
        for (int i = 0; i < NUM_ACTIONS; i++) {
            m_buf(i) = mask[i];
        }

        // Call Python function
        py::object result = py_fn(state_arr, mask_arr, p1_dist, p2_dist, turn);

        // Extract results - expects (probs_array, value_scalar)
        auto result_tuple = result.cast<py::tuple>();
        py::array_t<float> probs_arr = result_tuple[0].cast<py::array_t<float>>();
        value_out = result_tuple[1].cast<float>();

        // Copy probs to output buffer
        auto probs_buf = probs_arr.unchecked<1>();
        for (int i = 0; i < NUM_ACTIONS; i++) {
            probs_out[i] = probs_buf(i);
        }
    };
}

PYBIND11_MODULE(quoridor_core, m) {
    m.doc() = "C++ Quoridor engine with MCGS search";

    // ---------- QuoridorBoard ----------
    py::class_<QuoridorBoard>(m, "QuoridorBoard")
        .def(py::init<>())
        .def("reset", &QuoridorBoard::reset)

        .def("get_legal_actions", [](QuoridorBoard& self) {
            auto bits = self.get_legal_actions();
            py::array_t<bool> arr(NUM_ACTIONS);
            auto buf = arr.mutable_unchecked<1>();
            for (int i = 0; i < NUM_ACTIONS; i++) {
                buf(i) = bits.test(i);
            }
            return arr;
        })

        .def("get_state", [](QuoridorBoard& self) {
            float state[STATE_CHANNELS * STATE_SPATIAL * STATE_SPATIAL];
            std::bitset<NUM_ACTIONS> mask;
            float p1_dist, p2_dist;
            self.get_state(state, mask, p1_dist, p2_dist);

            py::array_t<float> state_arr(
                {STATE_CHANNELS, STATE_SPATIAL, STATE_SPATIAL},
                {STATE_SPATIAL * STATE_SPATIAL * sizeof(float),
                 STATE_SPATIAL * sizeof(float),
                 sizeof(float)}
            );
            auto buf = state_arr.mutable_unchecked<3>();
            int idx = 0;
            for (int ch = 0; ch < STATE_CHANNELS; ch++) {
                for (int r = 0; r < STATE_SPATIAL; r++) {
                    for (int c = 0; c < STATE_SPATIAL; c++) {
                        buf(ch, r, c) = state[idx++];
                    }
                }
            }

            py::array_t<bool> mask_arr(NUM_ACTIONS);
            auto mbuf = mask_arr.mutable_unchecked<1>();
            for (int i = 0; i < NUM_ACTIONS; i++) {
                mbuf(i) = mask.test(i);
            }

            // Return (state, mask, p1_dist, p2_dist)
            return py::make_tuple(state_arr, mask_arr, p1_dist, p2_dist);
        })

        .def("step", &QuoridorBoard::step)
        .def("is_terminal", &QuoridorBoard::is_terminal)
        .def("get_result", &QuoridorBoard::get_result,
             py::arg("perspective") = 0)
        .def("clone", [](const QuoridorBoard& self) {
            return QuoridorBoard(self);
        })
        .def("get_hash", &QuoridorBoard::get_hash)

        .def_property_readonly("turn", &QuoridorBoard::get_turn)
        .def_property_readonly("p1_pos", [](QuoridorBoard& self) {
            auto p = self.get_p1_pos();
            return py::make_tuple(p.first, p.second);
        })
        .def_property_readonly("p2_pos", [](QuoridorBoard& self) {
            auto p = self.get_p2_pos();
            return py::make_tuple(p.first, p.second);
        })
        .def_property_readonly("p1_walls", &QuoridorBoard::get_p1_walls)
        .def_property_readonly("p2_walls", &QuoridorBoard::get_p2_walls)
        .def("get_last_p1_dist", &QuoridorBoard::get_last_p1_dist)
        .def("get_last_p2_dist", &QuoridorBoard::get_last_p2_dist)
        .def("has_h_wall", [](QuoridorBoard& self, int r, int c) {
            return self.get_h_walls().test(r * 8 + c);
        })
        .def("has_v_wall", [](QuoridorBoard& self, int r, int c) {
            return self.get_v_walls().test(r * 8 + c);
        });

    // ---------- MCGS ----------
    py::class_<MCGS>(m, "MCGS")
        .def(py::init([](py::function eval_fn, float c_puct, float fpu_reduction) {
            return std::make_unique<MCGS>(make_eval_callback(eval_fn), c_puct, fpu_reduction);
        }), py::arg("eval_fn"), py::arg("c_puct") = 1.5f, py::arg("fpu_reduction") = 0.01f)

        .def("search", [](MCGS& self, QuoridorBoard& board, int num_simulations) {
            auto visits = self.search(board, num_simulations);
            py::array_t<float> arr(NUM_ACTIONS);
            auto buf = arr.mutable_unchecked<1>();
            for (int i = 0; i < NUM_ACTIONS; i++) {
                buf(i) = visits[i];
            }
            return arr;
        }, py::arg("board"), py::arg("num_simulations") = 100);
}
