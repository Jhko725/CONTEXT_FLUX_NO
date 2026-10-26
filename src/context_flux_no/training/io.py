import logging
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import more_itertools
import numpy as np
from hydra.utils import instantiate
from jaxtyping import PyTree
from omegaconf import OmegaConf
from orbax.checkpoint import v1 as ocp


log = logging.getLogger(__name__)

# Values the ICLR checkpoints stored in their model config that the current code no longer
# accepts, and what they meant when those models were trained: `bias_hyperinit` (misspelled in
# 10 of 12 configs, F1) fell through `HypernetworkHead` to the default initialization. The
# loaded weights make the choice irrelevant to the forward pass; the stored value is left in
# the checkpoint as evidence for the 0.7 audit.
# orbax <= 0.11 `save_pytree` wrote the tree under this checkpointable name; 0.12 writes
# `state` (`STATE_CHECKPOINTABLE_KEY`) and `load_pytree` became an alias of `load`, which
# looks for `state` unless told otherwise.
LEGACY_CHECKPOINTABLE = "pytree"

_LEGACY_CONFIG_FIXES: dict[tuple[str, Any], Any] = {
    ("hypernet_init", "bias_hyperinit"): "default",
}


def translate_legacy_config(model_config: dict[str, Any]) -> dict[str, Any]:
    """The stored model config with `_LEGACY_CONFIG_FIXES` applied (top level); a copy."""
    fixed = dict(model_config)
    for (key, legacy), current in _LEGACY_CONFIG_FIXES.items():
        if fixed.get(key) == legacy:
            log.warning(
                "legacy checkpoint config %s=%r read as %r (F1)", key, legacy, current
            )
            fixed[key] = current
    return fixed


def _infer_step(ckpt_dir: Path) -> int:
    """Given a checkpoint directory created by orbax.checkpoint consisting of a
    checkpoint from a single train step, determine and return the corresponding step
    number."""
    ckpt_step_dirs = filter(
        lambda p: p.is_dir() and p.stem.isnumeric(), ckpt_dir.iterdir()
    )
    try:
        return int(more_itertools.one(ckpt_step_dirs).stem)
    except ValueError:
        raise ValueError(
            """Given directory contains checkpoints from multiple steps, and thus step 
            cannot be inferred."""
        )


def load_model(
    loaddir: str | Path,
    step_number: int | None = None,
) -> PyTree:
    """Loads model from a checkpoint created by the `save_model` function.

    loaddir: Directory containing the checkpoints
    step_number: Number corresponding to the specific checkpoint
    """

    loaddir = Path(loaddir).resolve()
    if step_number is None:
        step_number = _infer_step(loaddir)

    with ocp.training.Checkpointer(loaddir) as mngr_load:
        model_config = translate_legacy_config(
            dict(mngr_load.root_metadata().custom_metadata)
        )
        model_backbone = instantiate(OmegaConf.create(model_config))

        template, rest = checkpoint_template(model_backbone)
        weights_load = mngr_load.load(
            step_number, template, checkpointable_name=LEGACY_CHECKPOINTABLE
        )

    return eqx.combine(weights_load, rest)


def checkpoint_template(model: PyTree) -> tuple[PyTree, PyTree]:
    """`(template, rest)`: the leaves the trainer saved (`eqx.is_array`, `trainer.py`) as the
    pytree orbax restores into, and everything else. The template must ask for *every* stored
    leaf: a stored array with no template leaf is read anyway, through the checkpoint's
    sharding file — which names the training GPU and fails on CPU — and then dropped.
    DISCO's `DropPath.p` (a `np.float64` from `np.linspace`) is such a leaf; orbax takes
    numpy scalars only as 0-d arrays, so those become `np.asarray`."""
    arrays, rest = eqx.partition(model, eqx.is_array)
    arrays = jax.tree.map(
        lambda x: np.asarray(x) if isinstance(x, np.generic) else x, arrays
    )
    return arrays, rest
