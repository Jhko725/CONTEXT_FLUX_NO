"""Type aliases shared across the package: scalar and array annotations, and
`FilterSpec`."""

from collections.abc import Callable
from typing import Any, TypeAlias

from jaxtyping import Array, Complex, Float, Int, PyTree


IntScalar: TypeAlias = Int[Array, ""]
FloatScalar: TypeAlias = Float[Array, ""]

FloatArray: TypeAlias = Float[Array, "..."]
ComplexArray: TypeAlias = Complex[Array, "..."]

"""
A filter specification. Typically used on a pytree to filter out certain subtrees.
Boolean values are treated as-is, while callables are called on each element of the
pytree. If the callable returns True, the element is kept, otherwise it is filtered out.
"""
FilterSpec: TypeAlias = PyTree[bool | Callable[[Any], bool]]

"""
A real number as Python gives it: a `float` or an `int`. PEP 484 lets `float` admit an
`int`, which static checkers follow and beartype does not, so a value that a YAML config
writes as `0` or `1`, such as an end of a domain span, is annotated with this alias.
"""
Real: TypeAlias = float | int
