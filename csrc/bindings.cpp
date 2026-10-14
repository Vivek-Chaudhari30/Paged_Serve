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

// csrc/paged_attention.cu
torch::Tensor paged_attention_v1(torch::Tensor query, torch::Tensor key_cache,
                                 torch::Tensor value_cache, torch::Tensor block_tables,
                                 torch::Tensor seq_lens, double scale);

// csrc/cache_kernels.cu
void reshape_and_cache(torch::Tensor key, torch::Tensor value, torch::Tensor key_cache,
                       torch::Tensor value_cache, torch::Tensor slot_mapping);
void copy_blocks(torch::Tensor kv_cache, torch::Tensor block_mapping);
void swap_blocks(torch::Tensor src, torch::Tensor dst, torch::Tensor block_mapping);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PagedServe CUDA kernels";
  m.def("add_one", &add_one,
        "Add one to every element of a float32 CUDA tensor. A build canary: it "
        "proves the toolchain, the pybind11 binding, and the torch::Tensor "
        "round-trip all work, independently of any attention math.");
  m.def("paged_attention_v1", &paged_attention_v1,
        "Decode attention read in place through block tables. query [n, heads, dim], "
        "caches [blocks, block_size, kv_heads, dim], block_tables int64 [n, max_blocks], "
        "seq_lens int32 [n]. Returns [n, heads, dim].");
  m.def("reshape_and_cache", &reshape_and_cache,
        "Scatter key/value [tokens, kv_heads, dim] into the paged caches at slot_mapping.");
  m.def("copy_blocks", &copy_blocks,
        "Copy whole blocks (src, dst) pairs within a [layers, 2, blocks, ...] cache.");
  m.def("swap_blocks", &swap_blocks,
        "Copy (src, dst) block pairs between a CUDA and a host [blocks, ...] tensor.");
}
