import pytest
import jax
import equinox as eqx
from context_flux_no.nn.hypernetwork import HypernetworkHead

def test_unknown_init_raises():
    target = eqx.nn.MLP(2, 2, 4, 1, key=jax.random.key(0))
    with pytest.raises(ValueError, match="Unknown hypernetwork initialization"):
        HypernetworkHead(target, in_size=8, num_blocks=2, initialization="bias_hyperinit", key=jax.random.key(1))