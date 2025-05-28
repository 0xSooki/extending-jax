# JAX Custom Operations with C++ and CUDA

A demonstration project showing how to extend JAX with custom C++ and CUDA operations using the XLA FFI (Foreign Function Interface). This project implements a simple mathematical operation `foo(a, b) = sum(a * (b + 1))` with both CPU and GPU kernels, complete with gradient support.

## Features

- ✅ **Custom JAX operations** with C++ and CUDA implementations
- ✅ **Automatic differentiation** support via `jax.grad`
- ✅ **CPU and GPU kernels** with automatic fallback
- ✅ **JAX transformations** (JIT, vmap, etc.) compatibility
- ✅ **Comprehensive test suite** with 20+ test cases
- ✅ **Scalar output** for gradient computation
- ✅ **Modern CMake build system** with scikit-build-core

## Mathematical Operation

The implemented operation computes:

```
f(a, b) = sum(a * (b + 1))
```

Where:

- `a`, `b` are input tensors of the same shape
- The output is a scalar (sum of element-wise products)
- Gradients: `∇_a f = b + 1`, `∇_b f = a`

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd permanent-boost

# Install dependencies
pip install -r requirements.txt

# Build and install the package
pip install -e .
```

### Basic Usage

```python
import jax.numpy as jnp
import jax
from sooki import foo

# Create input tensors (must be float32)
a = jnp.array([1.0, 2.0], dtype=jnp.float32)
b = jnp.array([3.0, 4.0], dtype=jnp.float32)

# Compute the operation
result = foo(a, b)
print(f"foo(a, b) = {result}")  # Output: 14.0

# Compute gradients
grad_fn = jax.grad(foo, argnums=0)
grad_a = grad_fn(a, b)
print(f"∇_a foo = {grad_a}")  # Output: [4.0, 5.0] = b + 1
```

### Advanced Usage

```python
# Works with JAX transformations
jit_foo = jax.jit(foo)
result = jit_foo(a, b)

# Vectorized operations
batch_a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
batch_b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)
batch_result = jax.vmap(foo)(batch_a, batch_b)

# Value and gradient simultaneously
value, grad = jax.value_and_grad(foo, argnums=0)(a, b)
```

## Project Structure

```
├── src/
│   ├── cpu_ops.hpp         # CPU kernel implementations
│   ├── gpu_ops.cc          # GPU FFI bindings
│   ├── kernels.cc.cu       # CUDA kernel implementations
│   ├── kernels.h           # GPU function declarations
│   ├── main.cpp            # CPU FFI bindings
│   └── sooki/
│       ├── __init__.py     # Package initialization
│       └── ops.py          # Python interface and custom VJP
├── tests/
│   └── test_foo.py         # Comprehensive test suite
├── CMakeLists.txt          # Build configuration
├── pyproject.toml          # Python package configuration
└── README.md               # This file
```

## Implementation Details

### Python Interface

- Custom VJP (Vector-Jacobian Product) implementation
- Automatic CPU/GPU dispatch based on availability
- Integration with JAX's transformation system

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/test_foo.py -v

# Run specific test categories
pytest tests/test_foo.py::TestFooGradients -v
pytest tests/test_foo.py::TestFooJAXTransformations -v
```

Test coverage includes:

- ✅ Basic functionality
- ✅ Gradient computation accuracy
- ✅ JAX transformations (JIT, vmap, value_and_grad)
- ✅ Error handling and edge cases
- ✅ Performance and consistency
- ✅ Mathematical correctness verification

## Requirements

### System Requirements

- Python 3.8+
- CMake 3.15+
- C++14 compatible compiler
- CUDA Toolkit (optional, for GPU support)

### Python Dependencies

- JAX >= 0.4.31
- JAXlib >= 0.4.31
- NumPy
- pybind11

### Build Dependencies

- scikit-build-core
- ninja (build system)

## GPU Support

GPU kernels are automatically compiled if CUDA is available. The package gracefully falls back to CPU-only mode if:

- CUDA toolkit is not installed
- No CUDA-capable GPU is detected
- CUDA compilation fails

Check GPU support:

```python
import sooki
print("GPU support:", hasattr(sooki, 'gpu_ops'))
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

## License

This project demonstrates JAX extension techniques and is provided for educational purposes.

## Acknowledgments

This project was inspired by and builds upon the excellent tutorial and examples from:

**[dfm/extending-jax](https://github.com/dfm/extending-jax)** - A comprehensive guide to extending JAX with custom operations

The original repository by Dan Foreman-Mackey provides foundational examples and best practices for JAX extensions that were instrumental in developing this project.

## References

- [JAX Custom Operations Guide](https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html)
- [XLA FFI Documentation](https://github.com/google/jax/tree/main/jaxlib/xla_extension)
- [JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
