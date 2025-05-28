#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <complex>
#include <vector>
#include <iostream>

namespace ffi = xla::ffi;

void FooFwdKernelCpu(const float *a, const float *b,
                     float *result, float *b_plus_1, int64_t n)
{
  float sum = 0.0f;
  for (int64_t i = 0; i < n; ++i)
  {
    b_plus_1[i] = b[i] + 1.0f;
    float c_i = a[i] * b_plus_1[i];
    sum += c_i;
  }
  *result = sum;
}

ffi::Error FooFwdCpuImpl(ffi::Buffer<ffi::F32> a,
                         ffi::Buffer<ffi::F32> b,
                         ffi::ResultBuffer<ffi::F32> result,
                         ffi::ResultBuffer<ffi::F32> b_plus_1,
                         int64_t n)
{
  FooFwdKernelCpu(
      reinterpret_cast<const float *>(a.typed_data()),
      reinterpret_cast<const float *>(b.typed_data()),
      reinterpret_cast<float *>(result->typed_data()),
      reinterpret_cast<float *>(b_plus_1->typed_data()),
      n);
  return ffi::Error::Success();
}

void FooBwdKernelCpu(const float *scalar_grad, const float *a,
                     const float *b_plus_1, float *a_grad,
                     float *b_grad, int64_t n)
{

  float grad_value = *scalar_grad;
  for (int64_t i = 0; i < n; ++i)
  {
    a_grad[i] = grad_value * b_plus_1[i];
    b_grad[i] = grad_value * a[i];
  }
}

ffi::Error FooBwdCpuImpl(ffi::Buffer<ffi::F32> scalar_grad,
                         ffi::Buffer<ffi::F32> a,
                         ffi::Buffer<ffi::F32> b_plus_1,
                         ffi::ResultBuffer<ffi::F32> a_grad,
                         ffi::ResultBuffer<ffi::F32> b_grad,
                         int64_t n)
{
  FooBwdKernelCpu(
      scalar_grad.typed_data(),
      a.typed_data(),
      b_plus_1.typed_data(),
      a_grad->typed_data(),
      b_grad->typed_data(),
      n);
  return ffi::Error::Success();
}
