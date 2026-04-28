import hydra
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from cell_segmentation.data.datamodule import DataModule
from cell_segmentation.factory import build_loss, build_model
from cell_segmentation.model.lightning import SegmentationModel


@hydra.main(version_base=None, config_path="../config", config_name="inference.yaml")
def inference(cfg: DictConfig) -> None:
    lit = SegmentationModel(
        model=build_model(cfg),
        loss_fn=build_loss(cfg),
        n_classes=cfg.model.n_classes,
    )
    datamodule = DataModule(
        datadir=cfg.datamodule.datadir,
        batch_size=cfg.datamodule.batch_size,
        train_subdir=cfg.datamodule.train_subdir,
        val_subdir=cfg.datamodule.val_subdir,
        test_subdir=cfg.datamodule.test_subdir,
    )
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=False,
    )
    trainer.test(model=lit, datamodule=datamodule, ckpt_path=cfg.checkpoint)


if __name__ == "__main__":
    inference()
