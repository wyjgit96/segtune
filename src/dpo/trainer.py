from __future__ import annotations

import os
import gc
import copy
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader

from src.model.trainer import Trainer
from src.dpo.dpo_dataset import DPODataset
from src.dpo.dpo_cfm import DPOCFM
from src.model.utils import exists, default


class DPOTrainer(Trainer):
    """
    DPO trainer that inherits from SFT Trainer but uses DPO-specific
    dataset, model, and training loop.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # Initialize beta_dpo before calling parent constructor
        self.beta_dpo = 2000.0

        # Update wandb config with DPO-specific parameters
        super().__init__(*args, **kwargs)

        # Update wandb config with DPO-specific parameters after parent initialization
        if hasattr(self.accelerator, 'trackers') and self.accelerator.trackers:
            for tracker in self.accelerator.trackers:
                if hasattr(tracker, 'run') and tracker.run is not None:
                    # Update wandb config with DPO-specific parameters
                    tracker.run.config.update({
                        "beta_dpo": self.beta_dpo,
                        "training_type": "DPO",
                    })

    def get_dataloader(self):
        """Override to use DPO dataset instead of regular diffusion dataset"""
        print("Setting up DPO dataset...")
        dpo_dataset = DPODataset(
            file_path=self.args.file_path,
            max_frames=self.args.max_frames,
            min_frames=self.args.min_frames,
            sampling_rate=self.args.sampling_rate,
            downsample_rate=self.args.downsample_rate,
            precision=self.precision
        )

        self.train_dataloader = DataLoader(
            dataset=dpo_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=dpo_dataset.custom_collate_fn,
            persistent_workers=True
        )

        print(f"DPO dataset created with {len(dpo_dataset)} samples")

    def train(self, resumable_with_seed: int = None):
        """DPO-specific training loop"""
        train_dataloader = self.train_dataloader

        start_step = self.load_checkpoint()
        global_step = start_step

        if resumable_with_seed > 0:
            orig_epoch_step = len(train_dataloader)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if resumable_with_seed > 0 and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"DPO Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                    smoothing=0.15
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"DPO Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    smoothing=0.15
                )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    # Extract DPO batch components
                    text_inputs = batch["lrc"]
                    win_latent = batch["win_latent"].permute(0, 2, 1)  # (batch, seq_len, dim)
                    loss_latent = batch["loss_latent"].permute(0, 2, 1)  # (batch, seq_len, dim)
                    style_prompt = batch["prompt"].permute(0, 2, 1)
                    start_time = batch["start_time"]

                    # GT latent for SFT (optional)
                    gt_latent = None
                    if "gt_latent" in batch and batch["gt_latent"] is not None:
                        gt_latent = batch["gt_latent"].permute(0, 2, 1)

                    # DPO forward pass
                    loss, dpo_loss, raw_model_loss, raw_ref_loss, implicit_acc, diff_diff = self.model(
                        win_latent=win_latent,
                        loss_latent=loss_latent,
                        text=text_inputs,
                        style_prompt=style_prompt,
                        start_time=start_time,
                        beta_dpo=self.beta_dpo,
                        gt_latent=gt_latent
                    )

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.is_main:
                    self.ema_model.update()

                global_step += 1

                # Enhanced logging for DPO
                if self.accelerator.is_local_main_process:
                    log_dict = {
                        "loss": loss.item(),
                        "dpo_loss": dpo_loss.item(),
                        "raw_model_loss": raw_model_loss.item(),
                        "raw_ref_loss": raw_ref_loss.item(),
                        "implicit_acc": implicit_acc.item(),
                        "diff_diff": diff_diff.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                    self.accelerator.log(log_dict, step=global_step)

                # Update progress bar with DPO-specific metrics
                progress_bar.set_postfix({
                    "step": str(global_step),
                    "dpo_loss": f"{dpo_loss.item():.4f}",
                    "acc": f"{implicit_acc.item():.3f}"
                })

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    self.save_checkpoint(global_step)

                # if global_step % self.last_per_steps == 0:
                #     self.save_checkpoint(global_step, last=True)

        self.save_checkpoint(global_step, last=True)
        self.accelerator.end_training()

        if self.is_main:
            print(f"DPO training completed after {global_step} steps")
