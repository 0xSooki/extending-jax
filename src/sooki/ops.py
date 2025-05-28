from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import sooki

jax.config.update("jax_enable_x64", True)

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

    if a.size == 0:
        return jnp.array(0.0, dtype=a.dtype), (a, b)
    
    n = np.prod(a.shape).astype(np.int64)
    scalar_type = jax.ShapeDtypeStruct((), a.dtype)
    intermediate_type = jax.ShapeDtypeStruct(a.shape, a.dtype)

    def impl(target_name):
        return lambda: jax.ffi.ffi_call(
            target_name, (scalar_type, intermediate_type), vmap_method="sequential"
        )(a, b, n=n)

    result, b_plus_1 = jax.lax.platform_dependent(
        cpu=impl("foo_fwd_cpu"), cuda=impl("foo_fwd")
    )
    return result, (a, b_plus_1)


def foo_bwd(res, c_grad):
    a, b_plus_1 = res

    if a.size == 0:
        return jnp.zeros_like(a), jnp.zeros_like(a)

    assert c_grad.dtype == a.dtype
    assert a.dtype == b_plus_1.dtype
    n = np.prod(a.shape).astype(np.int64)
    out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)

    def impl(target_name):
        return lambda: jax.ffi.ffi_call(
            target_name, (out_type, out_type), vmap_method="sequential"
        )(c_grad, a, b_plus_1, n=n)

    return jax.lax.platform_dependent(cpu=impl("foo_bwd_cpu"), cuda=impl("foo_bwd"))


@jax.custom_vjp
def foo(a, b):
    result, _ = foo_fwd(a, b)
    return result


foo.defvjp(foo_fwd, foo_bwd)
