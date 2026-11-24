"""Every selectable model config builds against every data config it is declared to pair with.

The census composes `scripts/training/configs/config.yaml` with one model and one data config
and instantiates the model on a shrunk grid. The base configs (`*_base`) are not selectable,
since they leave mandatory values for the configs that derive from them. A model config listed
in `DIMENSIONS` is built for one spatial dimension and pairs only with data of that dimension;
every other model config pairs with every data config.
"""

import itertools
from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "scripts/training/configs"
).resolve()

DIMENSIONS = {
    "dpot": 1,
    "fno_spatiotemporal": 1,
    "hyperfluxno_vit": 1,
    "dpot_2d_euler": 2,
    "dpot_2d_scalar": 2,
}


def _selectable_models() -> list[str]:
    return sorted(
        p.stem
        for p in (CONFIG_DIR / "model").glob("*.yaml")
        if not p.stem.endswith("_base")
    )


def _data_dimensions() -> dict[str, int]:
    return {
        p.stem: OmegaConf.load(p).num_spatial_dims
        for p in sorted((CONFIG_DIR / "data").glob("*.yaml"))
    }


def _pairs() -> list[tuple[str, str]]:
    data_dimensions = _data_dimensions()
    return [
        (model, data)
        for model, data in itertools.product(_selectable_models(), data_dimensions)
        if DIMENSIONS.get(model, data_dimensions[data]) == data_dimensions[data]
    ]


def test_dimension_table_names_existing_model_configs():
    """Every model config named in `DIMENSIONS` exists and is selectable."""
    assert set(DIMENSIONS) <= set(_selectable_models())


@pytest.mark.parametrize(("model", "data"), _pairs())
def test_model_config_builds_against_data_config(model: str, data: str):
    """The model config builds against the data config it is paired with."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config", overrides=[f"model={model}", f"data={data}"]
        )
    # A shrunk grid keeps the patch-based encoders cheap to build.
    if "encoder_kwargs" in cfg.model and "grid_size" in cfg.model.encoder_kwargs:
        cfg.model.encoder_kwargs.grid_size = [16] * cfg.data.num_spatial_dims
        cfg.model.encoder_kwargs.patch_size = [4] * cfg.data.num_spatial_dims
    hydra.utils.instantiate(cfg.model)
