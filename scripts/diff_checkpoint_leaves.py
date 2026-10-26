"""Compare the leaves stored in an ICLR checkpoint with the leaves `load_model` would ask
orbax for, so a stored array that the rebuilt model does not carry (restored, then dropped —
and read with the GPU sharding file, which fails on CPU) can be named.
Usage: JAX_PLATFORMS=cpu uv run python scripts/diff_checkpoint_leaves.py checkpoints/<data>/<Model>/<Loss>/seed=0/<ts>
"""

import re
import sys
from pathlib import Path

import equinox as eqx
import jax
from context_flux_no.training.io import _infer_step, translate_legacy_config
from hydra.utils import instantiate
from omegaconf import OmegaConf
from orbax.checkpoint import v1 as ocp


def _name(keypath) -> str:
    """`.blocks[0]['w']` and orbax's `['blocks']['0']['w']` both → `blocks/0/w`."""
    return "/".join(re.findall(r"[A-Za-z0-9_]+", jax.tree_util.keystr(keypath)))


def main(ckpt_dir: Path) -> None:
    with ocp.training.Checkpointer(ckpt_dir) as c:
        step = _infer_step(ckpt_dir)
        stored = {
            _name(k): type(v).__name__
            for k, v in jax.tree_util.tree_leaves_with_path(
                c.pytree_metadata(step).metadata
            )
        }
        model = instantiate(
            OmegaConf.create(
                translate_legacy_config(dict(c.root_metadata().custom_metadata))
            )
        )
    template, rest = eqx.partition(model, lambda x: isinstance(x, jax.Array))
    asked = {_name(k) for k, _ in jax.tree_util.tree_leaves_with_path(template)}
    rest_leaves = {
        _name(k): type(v).__name__ for k, v in jax.tree_util.tree_leaves_with_path(rest)
    }
    print(f"stored {len(stored)} leaves, template asks for {len(asked)}")
    for key in sorted(set(stored) - asked):
        print(
            f"stored, not asked: {key}  stored={stored[key]}  rebuilt={rest_leaves.get(key)}"
        )
    for key in sorted(asked - set(stored)):
        print(f"asked, not stored: {key}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
