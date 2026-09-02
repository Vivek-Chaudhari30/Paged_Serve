// The build canary.
//
// Deliberately the most boring kernel that can exist. Its whole purpose is to
// separate two failure modes that are miserable to debug together: "the CUDA
// toolchain, the pybind11 binding, and the torch::Tensor round-trip work" from
// "my attention math is correct". Phase 4 of the roadmap is explicit about
// getting this compiling and passing before a single line of attention exists,
// because fighting nvcc flags while also debugging an online softmax is how
// people abandon the phase.
//
// It stays in the repo after the real kernel lands. When paged_attention.cu
// stops building on some new card or toolkit, this answers "is it the
// environment or is it me?" in one command.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

__global__ void add_one_kernel(const float* __restrict__ input,
                               float* __restrict__ output, int64_t numel) {
  const int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (index < numel) {
    output[index] = input[index] + 1.0f;
  }
}

}  // namespace

torch::Tensor add_one(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "add_one expects a CUDA tensor, got ", input.device());
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "add_one expects float32, got ", input.scalar_type());

  // A non-contiguous input would make the flat indexing above read the wrong
  // elements rather than fail, so normalise instead of trusting the caller.
  const torch::Tensor source = input.contiguous();
  torch::Tensor output = torch::empty_like(source);

  // Run on the tensor's own device, not whatever happens to be current. On a
  // multi-GPU box those differ, and the symptom is a silent illegal access.
  const at::cuda::CUDAGuard guard(source.device());

  const int64_t numel = source.numel();
  if (numel == 0) {
    return output;
  }

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);

  // The current stream, so this composes with the caller's ordering rather than
  // silently serialising against it on the default stream.
  add_one_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      source.data_ptr<float>(), output.data_ptr<float>(), numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}
