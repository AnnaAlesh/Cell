import hydra
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from cell_detection.data.datamodule import DetectionDataModule
from cell_detection.model.lightning import DetectionModel
from cell_detection.model.modules import TinyYOLO
from cell_detection.utils.training import DetectionPlotCallback, flatten_cfg


@hydra.main(version_base=None, config_path="../config", config_name="train_detection.yaml")
def train(cfg: DictConfig) -> None:
    if cfg.seed is not None:
        L.seed_everything(cfg.seed, workers=True)

    anchors = [list(a) for a in cfg.data.anchors]
    backbone = TinyYOLO(
        num_classes=cfg.model.num_classes,
        num_anchors=len(anchors),
        in_channels=cfg.model.in_channels,
    )
    lit = DetectionModel(
        model=backbone,
        anchors=anchors,
        image_size=cfg.data.image_size,
        grid_size=cfg.data.grid_size,
        num_classes=cfg.model.num_classes,
        nms_iou_threshold=cfg.nms_iou_threshold,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    datamodule = DetectionDataModule(
        datadir=cfg.data.datadir,
        anchors=anchors,
        image_size=cfg.data.image_size,
        grid_size=cfg.data.grid_size,
        ignore_iou_thresh=cfg.data.ignore_iou_thresh,
        batch_size=cfg.data.batch_size,
        train_subdir=cfg.data.train_subdir,
        val_subdir=cfg.data.val_subdir,
        test_subdir=cfg.data.test_subdir,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val/map",
            mode="max",
            dirpath=f"{cfg.output_dir}/checkpoints",
            filename="epoch_{epoch:03d}",
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        DetectionPlotCallback(nms_iou_threshold=cfg.nms_iou_threshold),
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
