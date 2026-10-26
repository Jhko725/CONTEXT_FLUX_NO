"""Key hygiene: no PRNG key is consumed twice while constructing a model (findings F5).

`jax.random.split(key, n)` followed by `keys[n]` silently clamps to `keys[n-1]` in JAX, and a
key handed to two submodules initializes them with correlated weights. Both happened in the ICLR
code (DISCO, HyperNeuralOperator). This test records every key passed to `jax.random.split`,
`fold_in` and the common samplers during `__init__` and fails on any repeat. Models are built
from their Hydra configs on tiny grids so that the test exercises the real constructors.
"""

from __future__ import annotations

from pathlib import Path

import jax
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CFG_DIR = (Path(__file__).resolve().parents[2] / "scripts/training/configs").resolve()

# (model config, data config) pairs that must build; grid sizes are shrunk below.
MODELS = [
    ("hyperfluxno", "cubic_1d"),
    ("hyperno_fno", "cubic_1d"),
    ("hyperno_unet", "cubic_1d"),
    ("hyperfluxno_2d_euler_v2", "euler_2d"),
    pytest.param(
        "disco",
        "cubic_1d",
        marks=pytest.mark.xfail(
            strict=True,
            reason="F15: the ICLR DISCO checkpoints need the operatornet's original "
            "(key-reusing) initialization; fixed with checkpointed operatornet_scale",
        ),
    ),
    ("dpot", "cubic_1d"),
]

SAMPLERS = ("normal", "uniform", "truncated_normal", "bernoulli", "orthogonal")


def _key_id(key) -> tuple | None:
    """Concrete key -> hashable id; None for traced keys (under vmap/jit), which cannot be compared."""
    if isinstance(key, jax.core.Tracer):
        return None
    try:
        if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
            key = jax.random.key_data(key)
        return tuple(int(x) for x in jax.device_get(key).ravel())
    except (jax.errors.JAXTypeError, TypeError):
        return None


@pytest.fixture
def key_log(monkeypatch):
    """Patch jax.random so every consumed key is recorded; yields the list of ids."""
    seen: list[tuple] = []
    for name in ("split", "fold_in", *SAMPLERS):
        original = getattr(jax.random, name)

        def wrapped(key, *args, _orig=original, **kwargs):
            kid = _key_id(key)
            if kid is not None:
                seen.append(kid)
            return _orig(key, *args, **kwargs)

        monkeypatch.setattr(jax.random, name, wrapped)
    return seen


def _build(model_name: str, data_name: str, seed: int = 0):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(
            config_name="config", overrides=[f"model={model_name}", f"data={data_name}"]
        )
    OmegaConf.set_struct(cfg, False)
    cfg.model.key.seed = seed
    ek = cfg.model.get("encoder_kwargs")
    if ek is not None and "grid_size" in ek:
        ek.grid_size = [16] * cfg.data.num_spatial_dims
        ek.patch_size = [4] * cfg.data.num_spatial_dims
    import hydra

    return hydra.utils.instantiate(cfg.model)


@pytest.mark.parametrize(
    ("model_name", "data_name"),
    MODELS,
    ids=[getattr(m, "values", m)[0] for m in MODELS],
)
def test_no_key_is_consumed_twice(model_name, data_name, key_log):
    _build(model_name, data_name)
    duplicates = {k for k in key_log if key_log.count(k) > 1}
    assert not duplicates, (
        f"{len(duplicates)} PRNG key(s) consumed more than once during __init__"
    )


@pytest.mark.parametrize(
    ("model_name", "data_name"),
    [("hyperfluxno", "cubic_1d"), ("disco", "cubic_1d"), ("dpot", "cubic_1d")],
    ids=["hyperfluxno", "disco", "dpot"],
)
def test_seed_determinism(model_name, data_name):
    import equinox as eqx

    a, b, c = (_build(model_name, data_name, seed=s) for s in (0, 0, 1))
    assert eqx.tree_equal(eqx.filter(a, eqx.is_array), eqx.filter(b, eqx.is_array))
    assert not eqx.tree_equal(eqx.filter(a, eqx.is_array), eqx.filter(c, eqx.is_array))
