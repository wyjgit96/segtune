from jam.trainer import WebDatasetTrainer
from jam.dpo.dpo_dataset import DPODataset
from jam.utils import GradientTracker
from torch.utils.data import DataLoader
from tqdm import tqdm


class DPOTrainer(WebDatasetTrainer):
    """
    DPO trainer that inherits from WebDatasetTrainer but uses DPO-specific
    dataset, model, and training loop.
    """
    def __init__(self, *args, beta_dpo=2000, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_dpo = beta_dpo

    def get_dataloader(self):
        """Override to use DPO dataset instead of regular diffusion dataset"""
        print("Setting up DPO dataset...")
        train_dataset = DPODataset(**self.data_cfg.train_dataset)
        self.train_dataloader = DataLoader(
            train_dataset, 
            **self.data_cfg.train_dataloader, 
            collate_fn=train_dataset.custom_collate_fn
        )
        
        print(f"DPO dataset created with {len(train_dataset.dpo_pairs)} pairs")

    def train(self):
        """DPO-specific training loop"""
        start_step = self.load_checkpoint()
        global_step = start_step

        self.model.train()

        progress_bar = tqdm(
            range(start_step, self.max_steps),
            desc=f"DPO Training",
            unit="step",
            disable=not self.accelerator.is_local_main_process,
            initial=start_step,
            total=self.max_steps,
            smoothing=0.15
        )
        gradient_tracker = GradientTracker(self.model)

        for batch in self.train_dataloader:
            if global_step >= self.max_steps:
                break

            global_step += 1
            log_now = (global_step == 1) or (global_step % self.log_every == 0 and self.accelerator.is_local_main_process)

            with self.accelerator.accumulate(self.model):
                # Extract DPO batch components
                text_inputs = batch["lrc"]
                win_latent = batch["win_latent"].permute(0, 2, 1)  # (batch, seq_len, dim)
                loss_latent = batch["loss_latent"].permute(0, 2, 1)  # (batch, seq_len, dim)
                gt_latent = batch["gt_latent"].permute(0, 2, 1)  # (batch, seq_len, dim)
                style_prompt = batch["prompt"]
                start_time = batch["start_time"]
                duration_abs = batch["duration_abs"]
                duration_rel = batch["duration_rel"]

                # DPO forward pass
                loss, dpo_loss, raw_model_loss, raw_ref_loss, implicit_acc, diff_diff = self.model(
                    win_latent=win_latent,
                    loss_latent=loss_latent,
                    text=text_inputs,
                    style_prompt=style_prompt,
                    start_time=start_time,
                    duration_abs=duration_abs,
                    duration_rel=duration_rel,
                    beta_dpo=self.beta_dpo,
                    gt_latent=gt_latent
                )
                
                self.accelerator.backward(loss)

                if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if log_now:
                    gradient_tracker.store_params()
                    gradient_tracker.compute_grad_norm()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if self.is_main and self.use_ema:
                self.ema_model.update()

            # Enhanced logging for DPO
            if log_now:
                gradient_tracker.compute_update_norm()
                log_dict = {
                    "loss": loss.item(),
                    "dpo_loss": dpo_loss.item(),
                    "raw_model_loss": raw_model_loss.item(),
                    "raw_ref_loss": raw_ref_loss.item(),
                    "implicit_acc": implicit_acc.item(),
                    "diff_diff": diff_diff.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    "grad_norm": gradient_tracker.grad_norm,
                    "update_norm": gradient_tracker.update_norm,
                }
                self.accelerator.log(log_dict, step=global_step)

            # Update progress bar with DPO-specific metrics
            progress_bar.set_postfix({
                "step": str(global_step), 
                "dpo_loss": f"{dpo_loss.item():.4f}",
                "acc": f"{implicit_acc.item():.3f}"
            })
            progress_bar.update(1)

            if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                self.save_checkpoint(global_step)

            if global_step % self.last_per_steps == 0:
                self.save_checkpoint(global_step, last=True)

        progress_bar.close()
        self.save_checkpoint(global_step, last=True)
        self.accelerator.end_training()
        
        print(f"DPO training completed after {global_step} steps") 