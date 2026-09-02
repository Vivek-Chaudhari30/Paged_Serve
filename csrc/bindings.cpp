// pybind11 module definition for the PagedServe CUDA extension.
//
// Declarations only. Every kernel lives in its own translation unit so a
// compile error points at one file, and so the module can grow without this
// file becoming the place every change collides.
//
// The module is named `pagedserve._C`. Nothing in the Python package imports it
// eagerly: `import pagedserve` must succeed on a laptop with no nvcc
// (AGENTS.md section 4.3), so the load goes through pagedserve/extension.py,
// which fails soft and falls back to the gather backend.

#include <torch/extension.h>

// csrc/trivial.cu
torch::Tensor add_one(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PagedServe CUDA kernels";
  m.def("add_one", &add_one,
        "Add one to every element of a float32 CUDA tensor. A build canary: it "
        "proves the toolchain, the pybind11 binding, and the torch::Tensor "
        "round-trip all work, independently of any attention math.");
}
