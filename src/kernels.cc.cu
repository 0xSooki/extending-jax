#include "kernels.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
__global__ void FooFwdKernel(const float *a, const float *b, float *result,
                             float *b_plus_1, int64_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;

  float local_sum = 0.0f;

  for (size_t i = tid; i < n; i += grid_stride)
  {
    b_plus_1[i] = b[i] + 1.0f;
    float c_i = a[i] * b_plus_1[i];
    local_sum += c_i;
  }

  atomicAdd(result, local_sum);
}

ffi::Error FooFwdHost(cudaStream_t stream, ffi::Buffer<ffi::F32> a,
                      ffi::Buffer<ffi::F32> b, ffi::ResultBuffer<ffi::F32> result,
                      ffi::ResultBuffer<ffi::F32> b_plus_1, int64_t n)
{
  const int block_dim = 128;
  const int grid_dim = std::min(32, (int)((n + block_dim - 1) / block_dim));

  cudaMemsetAsync(result->typed_data(), 0, sizeof(float), stream);

  FooFwdKernel<<<grid_dim, block_dim, 0, stream>>>(
      a.typed_data(), b.typed_data(), result->typed_data(), b_plus_1->typed_data(),
      n);

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

__global__ void FooBwdKernel(const float *scalar_grad,
                             const float *a,
                             const float *b_plus_1,
                             float *a_grad,
                             float *b_grad,
                             int64_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;

  float grad_value = *scalar_grad;

  for (size_t i = tid; i < n; i += grid_stride)
  {
    a_grad[i] = grad_value * b_plus_1[i];
    b_grad[i] = grad_value * a[i];
  }
}

ffi::Error FooBwdHost(cudaStream_t stream,
                      ffi::Buffer<ffi::F32> scalar_grad,
                      ffi::Buffer<ffi::F32> a,
                      ffi::Buffer<ffi::F32> b_plus_1,
                      ffi::ResultBuffer<ffi::F32> a_grad,
                      ffi::ResultBuffer<ffi::F32> b_grad,
                      int64_t n)
{
  const int block_dim = 128;
  const int grid_dim = std::min(32, (int)((n + block_dim - 1) / block_dim));
  FooBwdKernel<<<grid_dim, block_dim, 0, stream>>>(
      scalar_grad.typed_data(), a.typed_data(), b_plus_1.typed_data(),
      a_grad->typed_data(), b_grad->typed_data(), n);
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}