from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import sooki

# Note: Using float32 to match FFI expectations
# jax.config.update("jax_enable_x64", True)

gpu = False
gpu_targets = {}
if hasattr(sooki, "gpu_ops"):
    try:
        gpu_targets = sooki.gpu_ops.foo()
        for name, target in gpu_targets.items():
            jax.ffi.register_ffi_target(name, target, platform="CUDA")
            gpu = True
    except (ImportError, AttributeError) as e:
        print(f"GPU support initialization failed: {e}")
        gpu = False
else:
    print("No GPU module found. Continuing with CPU support only.")

for name, target in sooki.registrations().items():
    jax.ffi.register_ffi_target(name, target)


def foo_fwd(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    n = np.prod(a.shape).astype(np.int64)
    scalar_type = jax.ShapeDtypeStruct((), a.dtype)  # scalar output
    intermediate_type = jax.ShapeDtypeStruct(a.shape, a.dtype)  # b_plus_1 shape

    # Use GPU if available, otherwise use CPU
    ffi_name = "foo_fwd" if gpu else "foo_fwd_cpu"

    result, b_plus_1 = jax.ffi.ffi_call(
        ffi_name, (scalar_type, intermediate_type), vmap_method="sequential"
    )(a, b, n=n)
    return result, (a, b_plus_1)


def foo_bwd(res, c_grad):
    a, b_plus_1 = res
    assert c_grad.dtype == a.dtype
    assert a.dtype == b_plus_1.dtype
    n = np.prod(a.shape).astype(np.int64)
    out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)

    # Use GPU if available, otherwise use CPU
    ffi_name = "foo_bwd" if gpu else "foo_bwd_cpu"

    # c_grad is now a scalar, pass it directly to the FFI function
    return jax.ffi.ffi_call(ffi_name, (out_type, out_type), vmap_method="sequential")(
        c_grad, a, b_plus_1, n=n
    )


@jax.custom_vjp
def foo(a, b):
    result, _ = foo_fwd(a, b)
    return result


foo.defvjp(foo_fwd, foo_bwd)
