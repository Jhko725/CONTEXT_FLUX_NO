import equinox as eqx
import jax
import pytest
from context_flux_no.nn.hypernetwork import HypernetworkHead


def _target():
    return eqx.nn.MLP(2, 2, 4, 1, key=jax.random.key(0))


def test_unknown_init_raises():
    # The ICLR configs spelled this with an underscore; it must now fail loudly (findings F1).
    with pytest.raises(ValueError, match="Unknown hypernetwork initialization"):
        HypernetworkHead(
            in_size=8,
            target_network=_target(),
            num_blocks=2,
            initialization="bias_hyperinit",
            key=jax.random.key(1),
        )


@pytest.mark.parametrize("initialization", ["default", "bias-hyperinit"])
def test_known_inits_build(initialization):
    head = HypernetworkHead(
        in_size=8,
        target_network=_target(),
        num_blocks=2,
        initialization=initialization,
        key=jax.random.key(1),
    )
    assert head.in_size == 8
