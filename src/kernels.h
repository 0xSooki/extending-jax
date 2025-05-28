#ifndef KERNELS_H_
#define KERNELS_H_

#include "xla/ffi/api/ffi.h"
#include <cuda_runtime_api.h>

namespace ffi = xla::ffi;

ffi::Error FooFwdHost(cudaStream_t stream, ffi::Buffer<ffi::F32> a,
                      ffi::Buffer<ffi::F32> b, ffi::ResultBuffer<ffi::F32> result,
                      ffi::ResultBuffer<ffi::F32> b_plus_1, int64_t n);

ffi::Error FooBwdHost(cudaStream_t stream,
                      ffi::Buffer<ffi::F32> scalar_grad,
                      ffi::Buffer<ffi::F32> a,
                      ffi::Buffer<ffi::F32> b_plus_1,
                      ffi::ResultBuffer<ffi::F32> a_grad,
                      ffi::ResultBuffer<ffi::F32> b_grad,
                      int64_t n);

#endif // KERNELS_H_