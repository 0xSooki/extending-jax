#include "cpu_ops.hpp"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <type_traits>
#include <utility>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace ffi = xla::ffi;

template <typename T>
py::capsule EncapsulateFfiHandler(T *fn)
{
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FooFwdCpu, FooFwdCpuImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>() // a
        .Arg<ffi::Buffer<ffi::F32>>() // b
        .Ret<ffi::Buffer<ffi::F32>>() // result (scalar)
        .Ret<ffi::Buffer<ffi::F32>>() // b_plus_1
        .Attr<int64_t>("n")           // n

); // b_plus_1;

// Creates symbol FooBwd with C linkage that can be loaded using Python ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FooBwdCpu, FooBwdCpuImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>() // scalar_grad
        .Arg<ffi::Buffer<ffi::F32>>() // a
        .Arg<ffi::Buffer<ffi::F32>>() // b_plus_1
        .Ret<ffi::Buffer<ffi::F32>>() // a_grad
        .Ret<ffi::Buffer<ffi::F32>>() // b_grad
        .Attr<int64_t>("n")           // n
);                                    // b_plus_1;

PYBIND11_MODULE(_core, m)
{
  m.doc() = R"pbdoc(
        Permanent boost core module
        -----------------------

        .. currentmodule:: permanent_boost

        .. autosummary::
           :toctree: _generate

           permanent
    )pbdoc";
  m.def("registrations", []()
        {
    py::dict registrations;
    registrations["foo_fwd_cpu"] = EncapsulateFfiHandler(FooFwdCpu);
    registrations["foo_bwd_cpu"] = EncapsulateFfiHandler(FooBwdCpu);
    return registrations; });

  m.attr("__version__") = "dev";
}
