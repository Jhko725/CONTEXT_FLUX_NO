"""Train on a dataset from The Well (HDF5), e.g. `euler_multi_quadrants_periodicBC`.

    uv run python scripts/training/train_the_well.py data=well_euler_2d model=hyperfluxno_2d_euler_v2

Uses `training=the_well` (gradient clipping in the optimizer chain) unless overridden. The run
lives in Hydra's output directory; see `training.run` and `train_multiphysics.py`.
"""

import grain
import hydra
from context_flux_no.data import TheWellDataSource
from context_flux_no.training import experiment as common
from omegaconf import DictConfig


def make_loader(source, *, batch_size: int, seed: int, worker_count: int):
    return grain.DataLoader(
        data_source=source,
        sampler=grain.samplers.IndexSampler(len(source), shuffle=True, seed=seed),
        operations=[grain.transforms.Batch(batch_size=batch_size)],
        worker_count=worker_count,
    )


@hydra.main(config_path="./configs", config_name="config_the_well", version_base=None)
def main(cfg: DictConfig) -> None:
    common.select_device(cfg)

    model = hydra.utils.instantiate(cfg.model)
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    window = common.segment_length(cfg, loss_fn)

    def source(split: str) -> TheWellDataSource:
        return TheWellDataSource(
            cfg.data.well_base_path,
            cfg.data.well_dataset_name,
            split,
            window_size=window,
            downsample_spatial=cfg.data.downsample_spatial,
            use_normalization=cfg.data.use_normalization,
        )

    source_train, source_valid = source("train"), source("valid")
    kwargs = dict(
        batch_size=cfg.training.batch_size, worker_count=cfg.data.worker_count
    )
    common.train(
        cfg,
        model=model,
        loss_fn=loss_fn,
        loss_args=(0.015, 1 / 64, 1 / 64),
        train_loader=make_loader(source_train, seed=0, **kwargs),
        valid_loader=make_loader(source_valid, seed=1, **kwargs),
        dataset_id=source_train.dataset_id,
    )


if __name__ == "__main__":
    main()
