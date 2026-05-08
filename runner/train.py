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
from collections.abc import Sequence
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from disco.model.disco import DISCO
from disco.training.bitnet import apply_bitnet_quantization
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
        optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer],
        base_lr: float,
        warmup_steps: int,
        decay_factor: float,
        decay_every_n_steps: int,
    ) -> None:
        self.optimizers = (
            list(optimizer) if isinstance(optimizer, Sequence) else [optimizer]
        )
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

        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
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
        self.validate_fp4_training_config()
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
        self.apply_training_quantization()

    def validate_fp4_training_config(self) -> None:
        fp4_cfg = self.configs.training.get("fp4", None)
        if fp4_cfg is None or not fp4_cfg.get("enabled", False):
            return

        fp4_format = fp4_cfg.get("format", "nvfp4")
        if fp4_format not in {"nvfp4", "mxfp4"}:
            raise ValueError("training.fp4.format must be 'nvfp4' or 'mxfp4'.")

        if not torch.cuda.is_available():
            raise RuntimeError("FP4 training requires a CUDA GPU.")

        capability = torch.cuda.get_device_capability(self.device)
        if capability[0] < 10:
            device_name = torch.cuda.get_device_name(self.device)
            raise RuntimeError(
                f"{fp4_format.upper()} training needs Blackwell-class native FP4 "
                f"support; current GPU is {device_name} with sm_{capability[0]}"
                f"{capability[1]}."
            )

        backend = fp4_cfg.get("backend", "transformer_engine")
        if backend != "transformer_engine":
            raise ValueError("training.fp4.backend currently supports only 'transformer_engine'.")
        if find_spec("transformer_engine") is None:
            raise ImportError(
                "training.fp4.enabled=true requires transformer_engine to be installed."
            )
        raise NotImplementedError(
            "FP4 module wrapping is not wired yet. The config gate is present so "
            "NVFP4/MXFP4 runs fail before model construction on unsupported setups."
        )

    def apply_training_quantization(self) -> None:
        bitnet_cfg = self.configs.training.get("bitnet", None)
        if bitnet_cfg is None or not bitnet_cfg.get("enabled", False):
            return
        report = apply_bitnet_quantization(self.model, bitnet_cfg)
        logger.info(
            "Enabled BitNet QAT: replaced %d linear modules covering %.2fM "
            "parameters; skipped %d modules.",
            report.replaced_modules,
            report.replaced_parameters / 1_000_000,
            report.skipped_modules,
        )

    def cast_trainable_parameters(self, dtype: torch.dtype) -> None:
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=dtype)

    def split_muon_parameters(
        self,
    ) -> tuple[list[tuple[str, torch.nn.Parameter]], list[torch.nn.Parameter]]:
        muon_params = []
        adam_params = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 2 and "embed" not in name.lower():
                muon_params.append((name, parameter))
            else:
                adam_params.append(parameter)
        return muon_params, adam_params

    def init_optimizer(self) -> None:
        opt_cfg = self.configs.training.optimizer
        optimizer_name = opt_cfg.get("name", "adam")
        cast_model_dtype = opt_cfg.get("cast_model_dtype")
        if cast_model_dtype is not None:
            if cast_model_dtype not in {"bf16", "fp16", "fp32"}:
                raise ValueError(
                    "training.optimizer.cast_model_dtype must be one of "
                    "null, 'bf16', 'fp16', or 'fp32'."
                )
            dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[cast_model_dtype]
            self.cast_trainable_parameters(dtype)

        params = [p for p in self.model.parameters() if p.requires_grad]
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                weight_decay=opt_cfg.weight_decay,
            )
            self.optimizers = [self.optimizer]
        elif optimizer_name in {"flash_adam", "flash_adamw"}:
            from flashoptim import FlashAdam, FlashAdamW

            optimizer_cls = FlashAdam if optimizer_name == "flash_adam" else FlashAdamW
            self.optimizer = optimizer_cls(
                params,
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                weight_decay=opt_cfg.weight_decay,
                quantize=opt_cfg.get("quantize", True),
                compress_state_dict=opt_cfg.get("compress_state_dict", True),
                master_weight_bits=opt_cfg.get("master_weight_bits", 24),
                fused=opt_cfg.get("fused", True),
            )
            self.optimizers = [self.optimizer]
        elif optimizer_name == "muon":
            from flashoptim import FlashAdam

            muon_params, adam_params = self.split_muon_parameters()
            if not muon_params:
                raise ValueError("Muon optimizer selected, but no 2D trainable params found.")
            logger.info(
                "Using Muon for %.2fM parameters and FlashAdam for %.2fM parameters.",
                sum(parameter.numel() for _, parameter in muon_params) / 1_000_000,
                sum(parameter.numel() for parameter in adam_params) / 1_000_000,
            )
            self.optimizer = torch.optim.Muon(
                muon_params,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.get("muon_weight_decay", opt_cfg.weight_decay),
                momentum=opt_cfg.get("muon_momentum", 0.95),
                nesterov=opt_cfg.get("muon_nesterov", True),
                ns_steps=opt_cfg.get("muon_ns_steps", 5),
                adjust_lr_fn=opt_cfg.get("muon_adjust_lr_fn", "match_rms_adamw"),
            )
            self.adam_optimizer = FlashAdam(
                adam_params,
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                weight_decay=opt_cfg.weight_decay,
                quantize=opt_cfg.get("quantize", True),
                compress_state_dict=opt_cfg.get("compress_state_dict", True),
                master_weight_bits=opt_cfg.get("master_weight_bits", 24),
                fused=opt_cfg.get("fused", True),
            )
            self.optimizers = [self.optimizer, self.adam_optimizer]
        else:
            raise ValueError(
                "training.optimizer.name must be one of 'adam', 'flash_adam', "
                "'flash_adamw', or 'muon'."
            )
        self.scheduler = WarmupStepDecay(
            optimizer=self.optimizers,
            base_lr=opt_cfg.lr,
            warmup_steps=self.configs.training.scheduler.warmup_steps,
            decay_factor=self.configs.training.scheduler.decay_factor,
            decay_every_n_steps=self.configs.training.scheduler.decay_every_n_steps,
        )
        setup_outputs = self.fabric.setup(self.model, *self.optimizers)
        self.model = setup_outputs[0]
        self.optimizers = list(setup_outputs[1:])
        self.optimizer = self.optimizers[0]
        self.scheduler.optimizers = self.optimizers
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

        if "optimizers" in checkpoint:
            optimizer_states = checkpoint["optimizers"]
            if len(optimizer_states) != len(self.optimizers):
                logger.warning(
                    "Skipping optimizer state load: checkpoint has %d optimizers, "
                    "current config has %d.",
                    len(optimizer_states),
                    len(self.optimizers),
                )
            else:
                for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
                    optimizer.load_state_dict(optimizer_state)
        elif "optimizer" in checkpoint:
            if len(self.optimizers) != 1:
                logger.warning(
                    "Skipping legacy optimizer state load for multi-optimizer config."
                )
            else:
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
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "ema": self.ema.state_dict(),
            "configs": OmegaConf.to_container(self.configs, resolve=True),
        }
        path = ckpt_dir / f"step_{step:07d}.pt"
        self.fabric.save(path, checkpoint)

    def _next_batch(self, dataloader_iter):
        try:
            return next(dataloader_iter), dataloader_iter
        except StopIteration:
            dataloader_iter = iter(self.dataloader)
            return next(dataloader_iter), dataloader_iter

    def train(self) -> None:
        self.model.train()
        start_step = getattr(self, "start_step", 0)
        max_steps = self.configs.training.max_steps
        gradient_accumulation_steps = int(
            self.configs.training.get("gradient_accumulation_steps", 1)
        )
        if gradient_accumulation_steps < 1:
            raise ValueError("training.gradient_accumulation_steps must be >= 1.")

        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        step = start_step
        dataloader_iter = iter(self.dataloader)
        while step < max_steps:
            lr = self.scheduler.step(step)
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            accumulated_logs: dict[str, torch.Tensor] = {}

            for micro_step in range(gradient_accumulation_steps):
                batch, dataloader_iter = self._next_batch(dataloader_iter)
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

                for key, value in log_dict.items():
                    detached = value.detach()
                    accumulated_logs[key] = accumulated_logs.get(key, 0.0) + detached

                sync_context = (
                    self.fabric.no_backward_sync(
                        self.model,
                        enabled=micro_step < gradient_accumulation_steps - 1,
                    )
                    if hasattr(self.fabric, "no_backward_sync")
                    else nullcontext()
                )
                with sync_context:
                    self.fabric.backward(loss / gradient_accumulation_steps)

            for optimizer in self.optimizers:
                self.fabric.clip_gradients(
                    self.model,
                    optimizer,
                    max_norm=self.configs.training.gradient_clip_norm,
                )
                optimizer.step()
            self.ema.update(self.model)

            if step % self.configs.training.log_every_n_steps == 0:
                log_values = {
                    key: float(
                        (value / gradient_accumulation_steps).detach().float().cpu()
                    )
                    for key, value in accumulated_logs.items()
                }
                log_values["lr"] = lr
                log_values["gradient_accumulation_steps"] = gradient_accumulation_steps
                self.fabric.log_dict(log_values, step=step)
                if self.fabric.is_global_zero:
                    logger.info(
                        "step=%d loss=%.4f lr=%.6g grad_accum=%d",
                        step,
                        log_values["loss/total"],
                        lr,
                        gradient_accumulation_steps,
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
