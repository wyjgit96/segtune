import torch
import torch.nn.functional as F
import copy
from random import random
from jam.model.cfm import CFM

class DPOCFM(CFM):
    """
    DPO (Direct Preference Optimization) version of CFM.
    
    This class modifies the CFM forward pass to support DPO training,
    where we compare win vs loss latent pairs using a reference model.
    """
    
    def __init__(self, sft='none', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert sft in ['none', 'win', 'gt'], f"Invalid SFT type: {sft}, must be one of ['none', 'win', 'gt']"
        self.sft = sft
        print(f"SFT type: {sft}")
        self.ref_transformer = None
        
    def setup_reference_model(self):
        """Create frozen reference model after initialization"""
        self.ref_transformer = copy.deepcopy(self.transformer)
        self.ref_transformer.requires_grad_(False)
        # Remove the .eval() calls - let training loop handle mode
        for param in self.ref_transformer.parameters():
            param.requires_grad = False
        print("Reference model set up for DPO training")
    
    def forward(self, win_latent, loss_latent, text, style_prompt=None, start_time=None, duration_abs=None, duration_rel=None, beta_dpo=None, gt_latent=None):
        """
        DPO Forward pass - takes win/loss latents separately
        
        Args:
            win_latent: Preferred latents -> shape (batch, seq_len, dim)
            loss_latent: Dispreferred latents -> shape (batch, seq_len, dim)
            text: Text tokens -> shape (batch, text_len)
            style_prompt: Style conditioning -> shape (batch, style_dim)
            start_time: Start time conditioning -> shape (batch,)
            duration_abs: Absolute duration conditioning -> shape (batch,)
            duration_rel: Relative duration conditioning -> shape (batch,)
            gt_latent: Ground truth latents -> shape (batch, seq_len, dim)
            
        Returns:
            dpo_loss: DPO loss scalar
            raw_model_loss: Raw model loss for logging
            raw_ref_loss: Raw reference loss for logging  
            implicit_acc: Implicit accuracy (how often model prefers win over loss)
        """
        # Lazy initialization of reference model
        if self.ref_transformer is None:
            self.setup_reference_model()

        if self.sft == 'gt' and gt_latent is None:
            raise ValueError("Ground truth latents are required for SFT")
            
        batch_size, seq_len, dim = win_latent.shape
        device = win_latent.device
        dtype = win_latent.dtype
        
        # Validate dimensions
        assert win_latent.shape == loss_latent.shape, f"Win and loss latents must have same shape, got {win_latent.shape} vs {loss_latent.shape}"
        
        # Concatenate win and loss latents for processing
        if self.sft == 'gt':
            inp = torch.cat([win_latent, loss_latent, gt_latent], dim=0)  # (3*batch, seq_len, dim)
        else:
            inp = torch.cat([win_latent, loss_latent], dim=0)  # (2*batch, seq_len, dim)
        
        repeat_num = 3 if self.sft == 'gt' else 2
        # Duplicate text and other conditioning for win/loss pairs
        text_duplicated = text.repeat(repeat_num, 1)  # (repeat_num*batch, text_len)
        style_prompt = style_prompt.repeat(repeat_num, 1)  # (repeat_num*batch, style_dim)
        start_time = start_time.repeat(repeat_num)  # (repeat_num*batch,)
        duration_abs = duration_abs.repeat(repeat_num)  # (repeat_num*batch,)
        duration_rel = duration_rel.repeat(repeat_num)  # (repeat_num*batch,)
        
        # Generate same noise for win/loss pairs (critical for fair comparison)
        x0_single = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
        x0 = x0_single.repeat(repeat_num, 1, 1)  # Same noise for both win and loss
        
        # Generate same timesteps for win/loss pairs (critical for fair comparison)
        time_single = torch.normal(mean=0, std=1, size=(batch_size,), device=device)
        time_single = torch.nn.functional.sigmoid(time_single)
        time = time_single.repeat(repeat_num)  # Same timesteps for both win and loss
        
        # Flow matching setup
        x1 = inp  # Target latents (win + loss concatenated)
        
        # Sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0
        
        # NO CONDITIONING MASK - DPO predicts entire sequence
        cond = torch.zeros_like(x1)  # No conditioning, predict everything
        
        if self.dual_drop_prob is not None:
            drop_prompt = random() < self.dual_drop_prob[0]
            drop_text = drop_prompt and (random() < self.dual_drop_prob[1])
        else:
            drop_text = random() < self.lrc_drop_prob
            drop_prompt = random() < self.style_drop_prob
        if self.no_cond_drop:
            drop_text = False
            drop_prompt = False

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if self.no_edit:
            drop_audio_cond = True
        
        # Main model prediction
        pred = self.transformer(
            x=φ, cond=cond, text=text_duplicated, time=time, 
            drop_audio_cond=drop_audio_cond, drop_text=drop_text, drop_prompt=drop_prompt,
            style_prompt=style_prompt, start_time=start_time, 
            duration_abs=duration_abs, duration_rel=duration_rel
        )
        
        # Compute model losses (per-sample)
        model_loss_unreduced = F.mse_loss(pred, flow, reduction="none")
        # Reduce over spatial dimensions (seq_len, dim) but keep batch dimension
        model_losses = model_loss_unreduced.mean(dim=list(range(1, len(model_loss_unreduced.shape))))
        
        # Split into win/loss losses
        if self.sft == 'gt':
            model_losses_w, model_losses_l, model_losses_gt = model_losses.chunk(3)
        else:
            model_losses_w, model_losses_l = model_losses.chunk(2)
        model_diff = model_losses_w - model_losses_l  # Want this to be negative (win < loss)
        
        # Reference model prediction (no gradients)
        with torch.no_grad():
            ref_pred = self.ref_transformer(
                x=φ, cond=cond, text=text_duplicated, time=time,
                drop_audio_cond=drop_audio_cond, drop_text=drop_text, drop_prompt=drop_prompt,
                style_prompt=style_prompt, start_time=start_time,
                duration_abs=duration_abs, duration_rel=duration_rel
            )
            
            ref_loss_unreduced = F.mse_loss(ref_pred, flow, reduction="none")
            ref_losses = ref_loss_unreduced.mean(dim=list(range(1, len(ref_loss_unreduced.shape))))
            if self.sft == 'gt':
                ref_losses_w, ref_losses_l, ref_losses_gt = ref_losses.chunk(3)
            else:
                ref_losses_w, ref_losses_l = ref_losses.chunk(2)
            ref_diff = ref_losses_w - ref_losses_l
        
        # DPO loss calculation (following TangoFlux exactly)
        scale_term = -0.5 * beta_dpo
        inside_term = scale_term * (model_diff - ref_diff)
        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
        
        # Final DPO loss: preference loss + regularization term
        # dpo_loss = -F.logsigmoid(inside_term).mean()
        dpo_loss = F.softplus(-inside_term).mean()
        if self.sft == 'gt':
            loss = dpo_loss + 0.2 * model_losses_gt.mean()
        elif self.sft == 'win':
            loss = dpo_loss + 0.2 * model_losses_w.mean()
        else:
            loss = dpo_loss
        
        # Return loss and metrics for logging
        if self.sft == 'gt':
            return loss, dpo_loss, model_losses_gt.mean(), ref_losses_gt.mean(), implicit_acc, (model_diff - ref_diff).mean()
        else:
            return loss, dpo_loss, model_losses_w.mean(), ref_losses_w.mean(), implicit_acc, (model_diff - ref_diff).mean()