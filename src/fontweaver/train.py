import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from fontweaver.config import base_config as cfg, ROOT_DIR
from fontweaver.model import FontweaverModel, FontweaverDataModule
from fontweaver.sampler import SamplingCallback
from pathlib import Path
import logging
from lightning.pytorch.profilers import AdvancedProfiler
import json
from dataclasses import asdict
import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main():
    logging.basicConfig(level=logging.DEBUG)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")
    model = (
        FontweaverModel()
        if cfg.checkpoint_path is None
        else FontweaverModel.load_from_checkpoint(
            cfg.checkpoint_path,
        )
    )
    data_module = FontweaverDataModule()

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=-1,
        every_n_epochs=4,
        monitor="val_loss",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    profiler = AdvancedProfiler(
        filename="profiler-report",
        dump_stats=True,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.device,
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        precision=cfg.precision,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            SamplingCallback(sample_every_epoch=2),
        ],
        devices=-1 if cfg.device == "cuda" else "auto",
        logger=TensorBoardLogger(ROOT_DIR / "logs", name=cfg.experiment_name),
        profiler=profiler,
    )

    # save config
    config_file = Path(trainer.log_dir) / "cfg.json"
    (Path(trainer.log_dir) / "samples").mkdir(exist_ok=True, parents=True)
    config_file.parent.mkdir(exist_ok=True, parents=True)
    with config_file.open("w") as fp:
        json.dump(asdict(cfg), fp, indent=4)
    logging.info(f"Config saved to {config_file}")

    trainer.fit(
        model,
        data_module,
        # ckpt_path=cfg.checkpoint_path,
    )

    logging.info("Training complete.")


if __name__ == "__main__":
    main()
