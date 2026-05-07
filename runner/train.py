# Copyright 2026 Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from disco.model.disco import DISCO
from disco.training.ema import ModelEma
from disco.training.step import compute_training_loss
from disco.utils.seed import seed_everything
from disco.utils.torch_utils import to_device
from runner.utils import print_config_tree

OmegaConf.register_new_resolver("gt", lambda a, b: a > b)
rootutils.setup_root(__file__, indicator=".project-root")

logger = logging.getLogger(__name__)


class WarmupStepDecay:
    """Linear warmup followed by multiplicative step decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_steps: int,
        decay_factor: float,
        decay_every_n_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.decay_every_n_steps = decay_every_n_steps

    def step(self, step: int) -> float:
        if step < self.warmup_steps:
            lr = self.base_lr * float(step + 1) / float(max(self.warmup_steps, 1))
        else:
            decay_steps = (step - self.warmup_steps) // self.decay_every_n_steps
            lr = self.base_lr * (self.decay_factor**decay_steps)

        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


class TrainRunner:
    """DISCO training runner scaffold.

    Dataset-specific code is intentionally injected through
    ``training.dataloader._target_`` so the PDB pipeline can be added without
    changing the model/loss/optimizer path.
    """

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
        self.load_checkpoint_if_requested()

    def init_env(self) -> None:
        strategy_name = self.configs.fabric.get("strategy", "auto")
        fabric_kwargs = {
            "num_nodes": self.configs.fabric.num_nodes,
            "loggers": [
                hydra.utils.instantiate(logger_cfg)
                for _, logger_cfg in self.configs.logger.items()
            ],
        }
        if strategy_name == "ddp":
            fabric_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
        elif strategy_name != "auto":
            fabric_kwargs["strategy"] = strategy_name

        self.fabric = Fabric(
            **fabric_kwargs,
        )
        self.fabric.launch()
        self.device = self.fabric.device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.9")

        if self.configs.deterministic:
            seed_everything(0, deterministic=True)

    def init_model(self) -> None:
        structure_encoder = None
        if self.configs.structure_encoder.use_structure_encoder:
            structure_encoder = hydra.utils.instantiate(
                self.configs.structure_encoder.args
            )

        sequence_sampling_strategy = hydra.utils.instantiate(
            self.configs.sequence_sampling_strategy
        )
        self.model = DISCO(
            self.configs,
            structure_encoder,
            sequence_sampling_strategy,
        )

    def init_optimizer(self) -> None:
        opt_cfg = self.configs.training.optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            params,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            weight_decay=opt_cfg.weight_decay,
        )
        self.scheduler = WarmupStepDecay(
            optimizer=self.optimizer,
            base_lr=opt_cfg.lr,
            warmup_steps=self.configs.training.scheduler.warmup_steps,
            decay_factor=self.configs.training.scheduler.decay_factor,
            decay_every_n_steps=self.configs.training.scheduler.decay_every_n_steps,
        )
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.ema = ModelEma(self.model, decay=self.configs.training.ema_decay)

    def init_dataloader(self) -> None:
        dataloader_cfg = self.configs.training.dataloader
        if dataloader_cfg is None:
            raise ValueError(
                "No training dataloader is configured yet. Add a Hydra target at "
                "`training.dataloader` that yields batches with `input_feature_dict`, "
                "`coordinate`, `coordinate_mask`, and `true_prot_restype`."
            )
        self.dataloader = hydra.utils.instantiate(dataloader_cfg)
        self.dataloader = self.fabric.setup_dataloaders(self.dataloader)

    def load_checkpoint_if_requested(self) -> None:
        checkpoint_path = self.configs.training.resume_checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = self.configs.load_checkpoint_path
        if checkpoint_path is None:
            return

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        model_state = checkpoint.get("model", checkpoint)
        sample_key = next(iter(model_state.keys()))
        if sample_key.startswith("module."):
            model_state = OrderedDict(
                (key[len("module.") :], value) for key, value in model_state.items()
            )
        self.unwrap_model().load_state_dict(
            model_state,
            strict=self.configs.load_strict,
        )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        self.start_step = int(checkpoint.get("step", 0))

    def unwrap_model(self) -> torch.nn.Module:
        model = self.model
        while hasattr(model, "module"):
            model = model.module
        return model

    def save_checkpoint(self, step: int) -> None:
        ckpt_dir = Path(self.configs.training.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "step": step,
            "model": self.unwrap_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "configs": OmegaConf.to_container(self.configs, resolve=True),
        }
        path = ckpt_dir / f"step_{step:07d}.pt"
        self.fabric.save(path, checkpoint)

    def train(self) -> None:
        self.model.train()
        start_step = getattr(self, "start_step", 0)
        max_steps = self.configs.training.max_steps
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        step = start_step
        while step < max_steps:
            for batch in self.dataloader:
                if step >= max_steps:
                    break

                lr = self.scheduler.step(step)
                batch = to_device(batch, self.device)
                enable_amp = (
                    torch.autocast(device_type="cuda", dtype=train_precision)
                    if torch.cuda.is_available() and self.configs.dtype != "fp32"
                    else nullcontext()
                )

                with enable_amp:
                    loss, log_dict = compute_training_loss(
                        self.model,
                        batch,
                        loss_weights=dict(self.configs.training.loss_weights),
                        sequence_conditioning_mode=(
                            self.configs.training.sequence_conditioning_mode
                        ),
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.fabric.backward(loss)
                self.fabric.clip_gradients(
                    self.model,
                    self.optimizer,
                    max_norm=self.configs.training.gradient_clip_norm,
                )
                self.optimizer.step()
                self.ema.update(self.model)

                if step % self.configs.training.log_every_n_steps == 0:
                    log_values = {
                        key: float(value.detach().float().cpu())
                        for key, value in log_dict.items()
                    }
                    log_values["lr"] = lr
                    self.fabric.log_dict(log_values, step=step)
                    if self.fabric.is_global_zero:
                        logger.info(
                            "step=%d loss=%.4f lr=%.6g",
                            step,
                            log_values["loss/total"],
                            lr,
                        )

                if step > 0 and step % self.configs.training.save_every_n_steps == 0:
                    self.save_checkpoint(step)

                step += 1

        self.save_checkpoint(step)


@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(configs: DictConfig) -> None:
    log_format = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    print_config_tree(configs, resolve=True)
    runner = TrainRunner(configs)
    runner.train()


if __name__ == "__main__":
    main()
