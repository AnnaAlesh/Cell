import hydra
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from cell_segmentation.data.datamodule import DataModule
from cell_segmentation.factory import build_loss, build_model
from cell_segmentation.model.lightning import SegmentationModel
from cell_segmentation.utils.training import PredictionPlotCallback, flatten_cfg


@hydra.main(version_base=None, config_path="../config", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    if cfg.seed is not None:
        L.seed_everything(cfg.seed, workers=True)

    lit = SegmentationModel(
        model=build_model(cfg),
        loss_fn=build_loss(cfg),
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        n_classes=cfg.model.n_classes,
    )

    datamodule = DataModule(
        datadir=cfg.datamodule.datadir,
        batch_size=cfg.datamodule.batch_size,
        train_subdir=cfg.datamodule.train_subdir,
        val_subdir=cfg.datamodule.val_subdir,
        test_subdir=cfg.datamodule.test_subdir,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            dirpath=f"{cfg.output_dir}/checkpoints",
            filename="epoch_{epoch:03d}",
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        PredictionPlotCallback(),
    ]
    if cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.early_stopping.monitor,
                patience=cfg.early_stopping.patience,
                mode=cfg.early_stopping.mode,
            )
        )
    logger = TensorBoardLogger(
        save_dir=f"{cfg.output_dir}/tensorboard", name=None, version=""
    )
    logger.log_hyperparams(flatten_cfg(cfg))

    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model=lit, datamodule=datamodule, ckpt_path=cfg.checkpoint)


if __name__ == "__main__":
    train()
