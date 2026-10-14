// Paged attention, v1: decode-only, one thread block per (head, sequence).
//
// Reads K and V *in place* through the block table. Nothing is gathered into a
// scratch buffer, which is the entire difference from the Python oracle in
// pagedserve/attention/gather.py and the reason this file exists.
//
// v1 is the "correct and slow" kernel the roadmap asks for, written to be
// read and diffed against the oracle rather than to be fast:
//
// - blockDim.x is head_dim rounded up to a power of two. Thread `d` owns
//   query[d] and the output accumulator acc[d] for the whole sequence.
// - Tokens are walked in order. Each token's score q.k is a shared-memory tree
//   reduction across the threads of the block.
// - Softmax is online (Milkovich & Gimelshein; FlashAttention): a running max
//   `m` and normaliser `l` are kept and the accumulator is rescaled whenever
//   the max grows, so the scores are never materialised and the kernel is a
//   single pass over the KV.
// - Accumulation is fp32 regardless of the storage dtype.
// - Grouped-query attention is handled by index arithmetic: query head `h`
//   reads KV head `h / (num_heads / num_kv_heads)`. K/V are never repeated.
//
// The obvious v2 work (a warp per token, vectorised 16-byte loads, splitting
// long sequences across thread blocks) changes nothing in the interface.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kMaxHeadDim = 256;

int next_power_of_two(int value) {
  int result = 32;  // never below a warp
  while (result < value) result <<= 1;
  return result;
}

template <typename scalar_t>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,                // [num_seqs, num_heads, head_dim]
    const scalar_t* __restrict__ query,        // [num_seqs, num_heads, head_dim] (strided)
    const scalar_t* __restrict__ key_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const scalar_t* __restrict__ value_cache,  // same layout as key_cache
    const int64_t* __restrict__ block_tables,  // [num_seqs, max_blocks]
    const int32_t* __restrict__ seq_lens,      // [num_seqs]
    const int num_kv_heads, const int head_dim, const int block_size, const int max_blocks,
    const int64_t q_stride_seq, const int64_t q_stride_head, const float scale) {
  extern __shared__ float reduce_buf[];  // blockDim.x floats

  const int head = blockIdx.x;
  const int seq = blockIdx.y;
  const int num_heads = gridDim.x;
  const int d = threadIdx.x;
  const bool active = d < head_dim;

  const int kv_head = head / (num_heads / num_kv_heads);
  const int seq_len = seq_lens[seq];
  const int64_t* table = block_tables + static_cast<int64_t>(seq) * max_blocks;

  const int64_t token_stride = static_cast<int64_t>(num_kv_heads) * head_dim;
  const int64_t block_stride = static_cast<int64_t>(block_size) * token_stride;
  const int64_t head_offset = static_cast<int64_t>(kv_head) * head_dim + d;

  // Fold the softmax scale into q once rather than into every score.
  const float q = active
                      ? static_cast<float>(query[seq * q_stride_seq + head * q_stride_head + d]) *
                            scale
                      : 0.0f;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    const int64_t block = table[t / block_size];
    const int64_t offset = block * block_stride + (t % block_size) * token_stride + head_offset;

    reduce_buf[d] = active ? q * static_cast<float>(key_cache[offset]) : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (d < stride) reduce_buf[d] += reduce_buf[d + stride];
      __syncthreads();
    }
    const float score = reduce_buf[0];
    // Every thread must read the score before the next token overwrites it.
    __syncthreads();

    const float new_max = fmaxf(running_max, score);
    // exp(-inf) == 0, so the first token needs no special case.
    const float correction = expf(running_max - new_max);
    const float p = expf(score - new_max);
    const float v = active ? static_cast<float>(value_cache[offset]) : 0.0f;
    running_sum = running_sum * correction + p;
    acc = acc * correction + p * v;
    running_max = new_max;
  }

  if (active) {
    const float result = running_sum > 0.0f ? acc / running_sum : 0.0f;
    out[(static_cast<int64_t>(seq) * num_heads + head) * head_dim + d] =
        static_cast<scalar_t>(result);
  }
}

}  // namespace

torch::Tensor paged_attention_v1(torch::Tensor query, torch::Tensor key_cache,
                                 torch::Tensor value_cache, torch::Tensor block_tables,
                                 torch::Tensor seq_lens, double scale) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda(), "KV cache must be on CUDA");
  TORCH_CHECK(block_tables.is_cuda() && seq_lens.is_cuda(),
              "block_tables and seq_lens must be on CUDA");
  TORCH_CHECK(query.dim() == 3, "query must be [num_seqs, num_heads, head_dim]");
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must be [num_blocks, block_size, num_kv_heads, head_dim]");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "K and V caches differ in shape");
  TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous(),
              "KV cache must be contiguous");
  TORCH_CHECK(query.scalar_type() == key_cache.scalar_type() &&
                  query.scalar_type() == value_cache.scalar_type(),
              "query and KV cache dtypes differ");
  TORCH_CHECK(query.stride(2) == 1, "query's last dimension must be contiguous");
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt64 && block_tables.dim() == 2,
              "block_tables must be a 2-D int64 tensor");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.dim() == 1,
              "seq_lens must be a 1-D int32 tensor");

  const int num_seqs = static_cast<int>(query.size(0));
  const int num_heads = static_cast<int>(query.size(1));
  const int head_dim = static_cast<int>(query.size(2));
  const int block_size = static_cast<int>(key_cache.size(1));
  const int num_kv_heads = static_cast<int>(key_cache.size(2));

  TORCH_CHECK(key_cache.size(3) == head_dim, "query and cache head_dim differ");
  TORCH_CHECK(head_dim <= kMaxHeadDim, "head_dim above ", kMaxHeadDim, " is not supported");
  TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads");
  TORCH_CHECK(block_tables.size(0) == num_seqs && seq_lens.size(0) == num_seqs,
              "block_tables / seq_lens batch size does not match query");

  auto out = torch::empty({num_seqs, num_heads, head_dim}, query.options());
  if (num_seqs == 0) return out;

  auto tables = block_tables.contiguous();
  auto lens = seq_lens.contiguous();

  const at::cuda::OptionalCUDAGuard guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = next_power_of_two(head_dim);
  const dim3 grid(num_heads, num_seqs);
  const size_t shared_bytes = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "paged_attention_v1",
      [&] {
        paged_attention_v1_kernel<scalar_t><<<grid, threads, shared_bytes, stream>>>(
            out.data_ptr<scalar_t>(), query.data_ptr<scalar_t>(), key_cache.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(), tables.data_ptr<int64_t>(),
            lens.data_ptr<int32_t>(), num_kv_heads, head_dim, block_size,
            static_cast<int>(tables.size(1)), query.stride(0), query.stride(1),
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
