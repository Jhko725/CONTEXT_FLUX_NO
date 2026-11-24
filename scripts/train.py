"""Train any model on any dataset; the data config's `source:` node picks the source class.

    uv run python scripts/train.py model=hyperfluxno data=cubic_1d
    uv run python scripts/train.py data=well_euler_2d training=the_well model=hyperfluxno_2d_euler_v2
    uv run python scripts/train.py ... wandb=online
    uv run python scripts/train.py resume=runs/2026-10-24/12-00-00
    uv run python scripts/train.py warm_start=runs/2026-10-24/12-00-00 model.key.seed=1

The run lives in Hydra's output directory (`runs/<date>/<time>`): `config.yaml`,
`metadata.json`, `checkpoints/`, `.hydra/` and (if enabled) `wandb/`. See `runs.run`.
"""

import grain
import hydra
import jax
from context_flux_no import experiment
from context_flux_no.data import WellDatasetSourceBase
from omegaconf import DictConfig


def make_loader(source: WellDatasetSourceBase, *, batch_size: int, seed: int):
    loader = (
        grain.MapDataset.source(source)
        .seed(seed)
        .shuffle()
        .repeat()
        .batch(batch_size=batch_size, drop_remainder=True)
        .to_iter_dataset(
            read_options=grain.ReadOptions(prefetch_buffer_size=20)
        )  # batches buffered on host
        .map(lambda x: jax.device_put(x, jax.extend.backend.get_default_device()))
    )
    return grain.experimental.ThreadPrefetchIterDataset(
        loader,
        prefetch_buffer_size=10,  # batches buffered on device
    )


@hydra.main(config_path="training/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    experiment.select_device(cfg)

    model = hydra.utils.instantiate(cfg.model)
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    window = experiment.segment_length(cfg, loss_fn)

    def source(split: str) -> WellDatasetSourceBase:
        return hydra.utils.instantiate(
            cfg.data.source, well_split_name=split, window_size=window
        )

    source_train, source_valid = source("train"), source("valid")
    experiment.train(
        cfg,
        model=model,
        loss_fn=loss_fn,
        loss_args=source_train.loss_args,
        train_loader=make_loader(
            source_train, batch_size=cfg.training.batch_size, seed=0
        ),
        valid_loader=make_loader(
            source_valid, batch_size=cfg.training.batch_size, seed=1
        ),
        dataset_id=source_train.dataset_id,
    )


if __name__ == "__main__":
    main()
