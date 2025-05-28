import pytest
import jax
import jax.numpy as jnp
import numpy as np
from sooki import foo


class TestFooBasic:
    """Test basic functionality of the foo function."""

    def test_foo_simple_case(self):
        """Test foo with simple 2x2 matrices."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

        result = foo(a, b)

        # foo computes sum(a * (b + 1))
        # a * (b+1) = [[1*6, 2*7], [3*8, 4*9]] = [[6, 14], [24, 36]]
        # sum = 6 + 14 + 24 + 36 = 80
        expected = 80.0

        assert isinstance(result, jax.Array)
        assert result.shape == ()  # scalar output
        assert jnp.isclose(result, expected)

    def test_foo_different_shapes(self):
        """Test foo with different input shapes."""
        # Test 1D arrays
        a1d = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        b1d = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
        result1d = foo(a1d, b1d)
        # a * (b+1) = [1*5, 2*6, 3*7] = [5, 12, 21], sum = 38
        assert jnp.isclose(result1d, 38.0)

        # Test 3x3 matrices
        a3x3 = jnp.ones((3, 3), dtype=jnp.float32)
        b3x3 = jnp.ones((3, 3), dtype=jnp.float32) * 2.0
        result3x3 = foo(a3x3, b3x3)
        # a * (b+1) = 1 * (2+1) = 3 for each element, sum = 9 * 3 = 27
        assert jnp.isclose(result3x3, 27.0)

        # Test larger matrix
        a_large = jnp.ones((5, 4), dtype=jnp.float32) * 0.5
        b_large = jnp.ones((5, 4), dtype=jnp.float32) * 1.5
        result_large = foo(a_large, b_large)
        # a * (b+1) = 0.5 * (1.5+1) = 0.5 * 2.5 = 1.25 for each element
        # sum = 20 * 1.25 = 25
        assert jnp.isclose(result_large, 25.0)

    def test_foo_data_types(self):
        """Test foo with different JAX data types."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        # Test float32 (currently the only supported type)
        a_f32 = a.astype(jnp.float32)
        b_f32 = b.astype(jnp.float32)
        result_f32 = foo(a_f32, b_f32)
        assert result_f32.dtype == jnp.float32
        assert jnp.isclose(result_f32, 80.0)

        # Note: float64 is not currently supported by the FFI implementation
        # The backend only supports float32 (F32) buffers

    def test_foo_edge_cases(self):
        """Test foo with edge cases."""
        # Test with zeros
        a_zero = jnp.zeros((2, 2), dtype=jnp.float32)
        b_zero = jnp.zeros((2, 2), dtype=jnp.float32)
        result_zero = foo(a_zero, b_zero)
        # a * (b+1) = 0 * (0+1) = 0 for all elements, sum = 0
        assert jnp.isclose(result_zero, 0.0)

        # Test with negative values
        a_neg = jnp.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=jnp.float32)
        b_neg = jnp.array([[-0.5, -1.5], [-2.5, -3.5]], dtype=jnp.float32)
        result_neg = foo(a_neg, b_neg)
        # a * (b+1) = [[-1*0.5, -2*(-0.5)], [-3*(-1.5), -4*(-2.5)]]
        #           = [[-0.5, 1.0], [4.5, 10.0]]
        # sum = -0.5 + 1.0 + 4.5 + 10.0 = 15.0
        assert jnp.isclose(result_neg, 15.0)

        # Test with single element
        a_single = jnp.array([[5.0]], dtype=jnp.float32)
        b_single = jnp.array([[3.0]], dtype=jnp.float32)
        result_single = foo(a_single, b_single)
        # a * (b+1) = 5 * (3+1) = 5 * 4 = 20
        assert jnp.isclose(result_single, 20.0)


class TestFooGradients:
    """Test gradient computation for the foo function."""

    def test_gradient_basic(self):
        """Test basic gradient computation."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

        # Test gradient w.r.t. a
        grad_a_fn = jax.grad(foo, argnums=0)
        grad_a = grad_a_fn(a, b)

        # Mathematical expectation: d/da sum(a * (b+1)) = b+1
        expected_grad_a = b + 1
        assert jnp.allclose(grad_a, expected_grad_a)
        assert grad_a.shape == a.shape

        # Test gradient w.r.t. b
        grad_b_fn = jax.grad(foo, argnums=1)
        grad_b = grad_b_fn(a, b)

        # Mathematical expectation: d/db sum(a * (b+1)) = a
        expected_grad_b = a
        assert jnp.allclose(grad_b, expected_grad_b)
        assert grad_b.shape == b.shape

    def test_gradient_different_shapes(self):
        """Test gradients with different input shapes."""
        # 1D case
        a1d = jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32)
        b1d = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

        grad_a1d = jax.grad(foo, argnums=0)(a1d, b1d)
        grad_b1d = jax.grad(foo, argnums=1)(a1d, b1d)

        assert jnp.allclose(grad_a1d, b1d + 1)
        assert jnp.allclose(grad_b1d, a1d)

        # 3x3 case
        a3x3 = jnp.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float32
        )
        b3x3 = jnp.array(
            [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]], dtype=jnp.float32
        )

        grad_a3x3 = jax.grad(foo, argnums=0)(a3x3, b3x3)
        grad_b3x3 = jax.grad(foo, argnums=1)(a3x3, b3x3)

        assert jnp.allclose(grad_a3x3, b3x3 + 1)
        assert jnp.allclose(grad_b3x3, a3x3)

    def test_gradient_zero_case(self):
        """Test gradients when inputs contain zeros."""
        a = jnp.array([[0.0, 1.0], [2.0, 0.0]], dtype=jnp.float32)
        b = jnp.array([[3.0, 0.0], [0.0, 4.0]], dtype=jnp.float32)

        grad_a = jax.grad(foo, argnums=0)(a, b)
        grad_b = jax.grad(foo, argnums=1)(a, b)

        assert jnp.allclose(grad_a, b + 1)
        assert jnp.allclose(grad_b, a)

    def test_jacobian_equivalence(self):
        """Test that Jacobian equals gradient for scalar output."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

        grad_a = jax.grad(foo, argnums=0)(a, b)
        # Note: jacobian may trigger deprecation warnings due to internal vmap usage
        # but should still work with the sequential vmap method we've configured
        jacobian_a = jax.jacobian(foo, argnums=0)(a, b)

        # For scalar output, Jacobian should equal gradient
        assert jnp.allclose(grad_a, jacobian_a)

    def test_gradient_numerical_stability(self):
        """Test gradient computation with very small and large values."""
        # Very small values
        a_small = jnp.array([[1e-6, 2e-6], [3e-6, 4e-6]], dtype=jnp.float32)
        b_small = jnp.array([[5e-6, 6e-6], [7e-6, 8e-6]], dtype=jnp.float32)

        grad_a_small = jax.grad(foo, argnums=0)(a_small, b_small)
        assert jnp.allclose(grad_a_small, b_small + 1, rtol=1e-5)

        # Large values
        a_large = jnp.array([[1e6, 2e6], [3e6, 4e6]], dtype=jnp.float32)
        b_large = jnp.array([[5e6, 6e6], [7e6, 8e6]], dtype=jnp.float32)

        grad_a_large = jax.grad(foo, argnums=0)(a_large, b_large)
        assert jnp.allclose(grad_a_large, b_large + 1, rtol=1e-5)


class TestFooErrorHandling:
    """Test error handling and edge cases."""

    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise appropriate errors."""
        a = jnp.array([[1.0, 2.0]], dtype=jnp.float32)  # (1, 2)
        b = jnp.array([[3.0], [4.0]], dtype=jnp.float32)  # (2, 1)

        with pytest.raises(AssertionError):
            foo(a, b)

    def test_dtype_mismatch_error(self):
        """Test that mismatched dtypes raise appropriate errors."""
        a = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
        b = jnp.array([[3.0, 4.0]], dtype=jnp.float64)

        with pytest.raises(AssertionError):
            foo(a, b)

    def test_empty_array_handling(self):
        """Test behavior with empty arrays."""
        a_empty = jnp.array([], dtype=jnp.float32).reshape(0, 2)
        b_empty = jnp.array([], dtype=jnp.float32).reshape(0, 2)

        result = foo(a_empty, b_empty)
        assert jnp.isclose(result, 0.0)  # sum of empty array should be 0


class TestFooJAXTransformations:
    """Test foo function with various JAX transformations."""

    def test_vmap_transformation(self):
        """Test foo with vmap (vectorization)."""
        # Create batch of inputs
        a_batch = jnp.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=jnp.float32
        )
        b_batch = jnp.array(
            [[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float32
        )

        # Apply vmap - this will use sequential method we configured
        foo_vmapped = jax.vmap(foo)
        results = foo_vmapped(a_batch, b_batch)

        # Check that we get a batch of scalar results
        assert results.shape == (2,)

        # Verify each result manually
        result0 = foo(a_batch[0], b_batch[0])
        result1 = foo(a_batch[1], b_batch[1])

        assert jnp.isclose(results[0], result0)
        assert jnp.isclose(results[1], result1)

    def test_jit_compilation(self):
        """Test foo with JIT compilation."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

        # JIT compile the function
        foo_jitted = jax.jit(foo)

        # Test that JIT version gives same result
        result_normal = foo(a, b)
        result_jitted = foo_jitted(a, b)

        assert jnp.isclose(result_normal, result_jitted)

        # Test that gradients work with JIT
        grad_jitted = jax.jit(jax.grad(foo, argnums=0))
        grad_normal = jax.grad(foo, argnums=0)(a, b)
        grad_jitted_result = grad_jitted(a, b)

        assert jnp.allclose(grad_normal, grad_jitted_result)

    def test_value_and_grad(self):
        """Test foo with value_and_grad transformation."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

        # Test value_and_grad
        value_and_grad_fn = jax.value_and_grad(foo, argnums=0)
        value, grad = value_and_grad_fn(a, b)

        # Compare with separate computations
        expected_value = foo(a, b)
        expected_grad = jax.grad(foo, argnums=0)(a, b)

        assert jnp.isclose(value, expected_value)
        assert jnp.allclose(grad, expected_grad)


class TestFooPerformance:
    """Test performance-related aspects."""

    def test_large_inputs(self):
        """Test foo with larger input matrices."""
        # Test with reasonably large matrices
        size = 100
        a_large = jnp.ones((size, size), dtype=jnp.float32) * 0.1
        b_large = jnp.ones((size, size), dtype=jnp.float32) * 0.2

        result = foo(a_large, b_large)

        # Expected: sum(0.1 * (0.2 + 1)) = sum(0.1 * 1.2) = 10000 * 0.12 = 1200
        expected = size * size * 0.1 * 1.2
        # Use larger tolerance due to float32 precision limitations with large numbers
        assert jnp.isclose(
            result, expected, rtol=1e-3
        ), f"Expected {expected}, got {result}"

        # Test that gradients still work
        grad_a = jax.grad(foo, argnums=0)(a_large, b_large)
        assert jnp.allclose(grad_a, b_large + 1)

    def test_consistency_across_runs(self):
        """Test that foo gives consistent results across multiple runs."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

        # Run multiple times and check consistency
        results = [foo(a, b) for _ in range(10)]

        # All results should be identical
        for result in results[1:]:
            assert jnp.isclose(results[0], result)


class TestFooLimitations:
    """Test known limitations of the foo function."""

    def test_hessian_limitation(self):
        """Test that Hessian computation fails as expected due to FFI limitations."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

        # Hessian should fail with FFI limitation error
        with pytest.raises(
            ValueError, match="The FFI call .* cannot be differentiated"
        ):
            hessian_fn = jax.hessian(foo, argnums=0)
            hessian_fn(a, b)

    def test_unsupported_dtype_limitation(self):
        """Test that unsupported dtypes (like float64) fail appropriately."""
        # Currently, only float32 is supported by the FFI backend
        a_f64 = jnp.array([[1.0, 2.0]], dtype=jnp.float64)
        b_f64 = jnp.array([[3.0, 4.0]], dtype=jnp.float64)

        # This should fail with a buffer dtype error
        with pytest.raises(Exception):  # Could be XlaRuntimeError or similar
            foo(a_f64, b_f64)


class TestFooDocumentation:
    """Test documentation and examples from docstrings."""

    def test_readme_example(self):
        """Test the basic example that would appear in README."""
        # Basic usage example
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

        # Forward pass: computes sum(a * (b + 1))
        result = foo(a, b)
        assert result == 80.0

        # Backward pass: compute gradients
        grad_fn = jax.grad(foo, argnums=0)
        gradient = grad_fn(a, b)
        expected_gradient = b + 1  # Analytical gradient
        assert jnp.allclose(gradient, expected_gradient)

        # Works with JAX transformations
        jitted_foo = jax.jit(foo)
        assert jnp.isclose(jitted_foo(a, b), result)


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running smoke test...")

    a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

    result = foo(a, b)
    print(f"foo({a}, {b}) = {result}")

    grad_a = jax.grad(foo, argnums=0)(a, b)
    print(f"∇_a foo = {grad_a}")

    print("Smoke test passed!")
