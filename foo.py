from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from sooki import foo

a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)
c = foo(a, b)
print(c)
