"""Instantiate every model config against every data config it can pair with, on tiny shapes.
Usage: uv run python scripts/check_configs.py
"""
import itertools, sys
from pathlib import Path
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CFG_DIR = Path("scripts/training/configs").resolve()
MODELS = sorted(p.stem for p in (CFG_DIR / "model").glob("*.yaml"))
DATAS = sorted(p.stem for p in (CFG_DIR / "data").glob("*.yaml"))

failures = []
with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
    for m, d in itertools.product(MODELS, DATAS):
        try:
            cfg = compose(config_name="config", overrides=[f"model={m}", f"data={d}"])
            # shrink anything shape-bearing so instantiation is cheap
            if "encoder_kwargs" in cfg.model and "grid_size" in cfg.model.encoder_kwargs:
                cfg.model.encoder_kwargs.grid_size = [16] * cfg.data.num_spatial_dims
                cfg.model.encoder_kwargs.patch_size = [4] * cfg.data.num_spatial_dims
            hydra.utils.instantiate(cfg.model)
            print(f"OK    model={m} data={d}")
        except Exception as e:  # noqa: BLE001 — we want the full census
            failures.append((m, d, f"{type(e).__name__}: {e}"))
            print(f"FAIL  model={m} data={d}  {type(e).__name__}: {str(e)[:120]}")
print(f"\n{len(failures)} failing combinations")
sys.exit(1 if failures else 0)