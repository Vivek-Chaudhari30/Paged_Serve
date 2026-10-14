// KV cache maintenance kernels: scatter new K/V into blocks, copy blocks for
// copy-on-write, and move blocks between device and host for swap preemption.
//
// Each has a pure-PyTorch twin in pagedserve/attention/paged_reference.py; the
// tests diff one against the other.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace {

template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,      // [num_tokens, num_kv_heads, head_dim]
    const scalar_t* __restrict__ value,    // [num_tokens, num_kv_heads, head_dim]
    scalar_t* __restrict__ key_cache,      // [num_blocks, block_size, num_kv_heads, head_dim]
    scalar_t* __restrict__ value_cache,    // same layout
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t token_numel) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slot_mapping[token];
  // A negative slot marks a token with nowhere to go. The engine routes
  // padding to the trash block instead, but the kernel stays safe either way.
  if (slot < 0) return;
  // Blocks are [block_size, heads, dim] and contiguous, so slot * token_numel
  // is the token's first element in the flattened cache.
  const int64_t src = token * token_numel;
  const int64_t dst = slot * token_numel;
  for (int64_t i = threadIdx.x; i < token_numel; i += blockDim.x) {
    key_cache[dst + i] = key[src + i];
    value_cache[dst + i] = value[src + i];
  }
}

template <typename scalar_t>
__global__ void copy_blocks_kernel(scalar_t* __restrict__ kv_cache,  // [L*2, num_blocks, ...]
                                   const int64_t* __restrict__ block_mapping,  // [pairs, 2]
                                   const int64_t num_blocks, const int64_t block_numel) {
  const int64_t layer_kv = blockIdx.x;
  const int64_t pair = blockIdx.y;
  const int64_t src_block = block_mapping[2 * pair];
  const int64_t dst_block = block_mapping[2 * pair + 1];
  scalar_t* base = kv_cache + layer_kv * num_blocks * block_numel;
  const scalar_t* src = base + src_block * block_numel;
  scalar_t* dst = base + dst_block * block_numel;
  for (int64_t i = threadIdx.x; i < block_numel; i += blockDim.x) {
    dst[i] = src[i];
  }
}

}  // namespace

void reshape_and_cache(torch::Tensor key, torch::Tensor value, torch::Tensor key_cache,
                       torch::Tensor value_cache, torch::Tensor slot_mapping) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda(), "key and value must be CUDA tensors");
  TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda(), "KV cache must be on CUDA");
  TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be a CUDA int64 tensor");
  TORCH_CHECK(key.dim() == 3 && key.sizes() == value.sizes(),
              "key and value must both be [num_tokens, num_kv_heads, head_dim]");
  TORCH_CHECK(key_cache.dim() == 4 && key_cache.sizes() == value_cache.sizes(),
              "KV cache must be [num_blocks, block_size, num_kv_heads, head_dim]");
  TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous(),
              "KV cache must be contiguous");
  TORCH_CHECK(key.size(1) == key_cache.size(2) && key.size(2) == key_cache.size(3),
              "token shape does not match the cache");
  TORCH_CHECK(key.scalar_type() == key_cache.scalar_type(), "dtype mismatch");
  TORCH_CHECK(slot_mapping.numel() == key.size(0), "one slot per token is required");

  const int64_t num_tokens = key.size(0);
  if (num_tokens == 0) return;
  auto k = key.contiguous();
  auto v = value.contiguous();
  auto slots = slot_mapping.contiguous();
  const int64_t token_numel = k.size(1) * k.size(2);

  const at::cuda::OptionalCUDAGuard guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = static_cast<int>(std::min<int64_t>(token_numel, 1024));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, key.scalar_type(), "reshape_and_cache",
      [&] {
        reshape_and_cache_kernel<scalar_t><<<num_tokens, threads, 0, stream>>>(
            k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), key_cache.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(), slots.data_ptr<int64_t>(), token_numel);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_blocks(torch::Tensor kv_cache, torch::Tensor block_mapping) {
  TORCH_CHECK(kv_cache.is_cuda() && kv_cache.is_contiguous(),
              "kv_cache must be a contiguous CUDA tensor");
  TORCH_CHECK(kv_cache.dim() == 6,
              "kv_cache must be [layers, 2, num_blocks, block_size, kv_heads, head_dim]");
  TORCH_CHECK(block_mapping.scalar_type() == torch::kInt64 && block_mapping.dim() == 2 &&
                  block_mapping.size(1) == 2,
              "block_mapping must be an int64 [pairs, 2] tensor of (src, dst)");
  const int64_t pairs = block_mapping.size(0);
  if (pairs == 0) return;
  auto mapping = block_mapping.to(kv_cache.device()).contiguous();

  const int64_t layers_kv = kv_cache.size(0) * kv_cache.size(1);
  const int64_t num_blocks = kv_cache.size(2);
  const int64_t block_numel = kv_cache.size(3) * kv_cache.size(4) * kv_cache.size(5);

  const at::cuda::OptionalCUDAGuard guard(device_of(kv_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(layers_kv, pairs);
  const int threads = static_cast<int>(std::min<int64_t>(block_numel, 1024));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, kv_cache.scalar_type(), "copy_blocks", [&] {
        copy_blocks_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            kv_cache.data_ptr<scalar_t>(), mapping.data_ptr<int64_t>(), num_blocks, block_numel);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swap_blocks(torch::Tensor src, torch::Tensor dst, torch::Tensor block_mapping) {
  // src and dst are [num_blocks, ...] on different devices (one CUDA, one
  // host). A host tensor should be pinned or the copy degrades to synchronous.
  TORCH_CHECK(src.is_cuda() || dst.is_cuda(), "one side of a swap must be on CUDA");
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "swap tensors must be contiguous");
  TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "swap dtype mismatch");
  TORCH_CHECK(src[0].numel() == dst[0].numel(), "swap block shapes differ");
  TORCH_CHECK(block_mapping.device().is_cpu() && block_mapping.scalar_type() == torch::kInt64 &&
                  block_mapping.dim() == 2 && block_mapping.size(1) == 2,
              "block_mapping must be a CPU int64 [pairs, 2] tensor of (src, dst)");

  cudaMemcpyKind kind;
  if (src.is_cuda() && dst.is_cuda()) {
    kind = cudaMemcpyDeviceToDevice;
  } else if (src.is_cuda()) {
    kind = cudaMemcpyDeviceToHost;
  } else {
    kind = cudaMemcpyHostToDevice;
  }

  const auto cuda_device = src.is_cuda() ? src.device() : dst.device();
  const at::cuda::OptionalCUDAGuard guard(cuda_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t block_bytes = src[0].numel() * src.element_size();
  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());
  auto mapping = block_mapping.contiguous();
  const int64_t* pairs = mapping.data_ptr<int64_t>();
  for (int64_t i = 0; i < mapping.size(0); ++i) {
    const int64_t s = pairs[2 * i];
    const int64_t d = pairs[2 * i + 1];
    C10_CUDA_CHECK(cudaMemcpyAsync(dst_ptr + d * block_bytes, src_ptr + s * block_bytes,
                                   block_bytes, kind, stream));
  }
}
