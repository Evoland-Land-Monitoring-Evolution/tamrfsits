#!/usr/bin/env python

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Training script for tamrfsits
"""
import logging
import os
from pathlib import Path
from typing import cast

import hydra
import pytorch_lightning as pl
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../hydra/", config_name="train.yaml")
def main(config: DictConfig):
    """
    Main hydra
    """

    if config.get("loglevel"):
        # Configure logging
        numeric_level = getattr(logging, config.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {config.loglevel}")

        logging.basicConfig(
            level=numeric_level,
            datefmt="%y-%m-%d %H:%M:%S",
            format="%(asctime)s :: %(levelname)s :: %(message)s",
        )

    # Change current working directory to checkpoints dir, to enable auto
    # restart checkpointing
    checkpoints_dir = os.path.join(
        config.original_work_dir, "checkpoints", config.name, config.label
    )
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(checkpoints_dir)
    logging.info("Working directory: " + checkpoints_dir)
    # configure logging at the root level of Lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # Apply seed if needed
    if config.get("seed"):
        pl.seed_everything(config.get("seed"), workers=True)

    if config.get("mat_mul_precision"):
        torch.set_float32_matmul_precision(config.get("mat_mul_precision"))

    # load samples
    data_module = hydra.utils.instantiate(config.datamodule.data_module)

    # Training module
    training_module = hydra.utils.instantiate(config.training_module.training_module)
    if config.start_from_checkpoint is not None:
        logging.info(f"Restoring generator state from {config.start_from_checkpoint}")
        checkpoint = torch.load(
            config.start_from_checkpoint,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        # For each registered modules, read back parameters
        for m in training_module.the_modules.keys():
            p = {
                k.split(".", maxsplit=2)[2]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith(f"the_modules.{m}")
            }
            if p:
                logging.info(f"Loading parameters for module {m} from checkpoint")
                try:
                    training_module.the_modules[m].load_state_dict(p)
                except Exception:
                    logging.info(
                        f"Can not load parameters for module {m}, model may have changed"
                    )
            else:
                logging.info(f"No parameters found for module {m} in checkpoint")
    # Define callbacks
    # (from https://github.com/ashleve/lightning-hydra-template/blob/
    # a4b5299c26468e98cd264d3b57932adac491618a/src/training_pipeline.py#L50)
    callbacks: list[pl.Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                logging.info("Instantiating callback <%s>", cb_conf._target_)
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    loggers: list[pl.loggers.logger.Logger] = []
    if "loggers" in config:
        for _, lg_conf in config.loggers.items():
            if "_target_" in lg_conf:
                logging.info("Instantiating logger <%s>", lg_conf._target_)
                loggers.append(hydra.utils.instantiate(lg_conf))

    nb_training_batches = len(data_module.train_dataloader())
    nb_validation_batches = len(data_module.val_dataloader())

    logging.info(
        "nb_training_batches=%s, nb_validation_batches=%s",
        str(nb_training_batches),
        str(nb_validation_batches),
    )

    trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    if config.use_learning_rate_finder:
        tuner = pl.tuner.Tuner(trainer)
        lr_finder = tuner.lr_find(training_module, data_module, max_lr=1e-2)
        assert lr_finder is not None
        fig = cast(Figure, lr_finder.plot(suggest=True))
        fig.savefig(os.path.join(checkpoints_dir, "learning_rate.pdf"), format="pdf")
        logging.info(f"Suggested learning rate: {lr_finder.suggestion()}")

    if config.resume_from_checkpoint is None:
        # Fit the network
        trainer.fit(training_module, data_module)
    else:
        trainer.fit(
            training_module, data_module, ckpt_path=config.resume_from_checkpoint
        )


if __name__ == "__main__":
    main()
