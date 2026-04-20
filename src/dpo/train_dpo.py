import json
import os
from importlib.resources import files

import torch
from torch.utils.data import DataLoader
from prefigure.prefigure import get_all_args

from src.model import TemporalControlDiT
from src.dpo.dpo_cfm import DPOCFM
from src.dpo.trainer import DPOTrainer
from infer.infer_utils import load_checkpoint

os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"


def main():
    # import pdb
    # pdb.set_trace()
    args = get_all_args("config/default.ini")

    with open(args.model_config) as f:
        model_config = json.load(f)

    if model_config["model_type"] == "diffrhythm":
        wandb_resume_id = None
        model_cls = TemporalControlDiT

    # 使用 DPOCFM 替代 CFM
    model = DPOCFM(
        transformer=model_cls(**model_config["model"], max_frames=args.max_frames),
        num_channels=model_config["model"]['mel_dim'],
        audio_drop_prob=args.audio_drop_prob,
        style_drop_prob=args.style_drop_prob,
        lrc_drop_prob=args.lrc_drop_prob,
        max_frames=args.max_frames,
        sft='none',
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # 加载预训练权重
    if hasattr(args, 'pretrained_ckpt_path') and args.pretrained_ckpt_path.strip():
        print(f"Loading pretrained weights from: {args.pretrained_ckpt_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_checkpoint(
            model,
            args.pretrained_ckpt_path,
            device=device,
            use_ema=False,
            fp16=False,
            no_cond_encoder=not args.continue_train,
        )

    trainer = DPOTrainer(
        model,
        args,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        checkpoint_path=f"ckpts/{args.exp_name}",
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        wandb_project="diffrhythm-dpo",
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=args.last_per_steps,
        bnb_optimizer=False,
        reset_lr=args.reset_lr,
        batch_size=args.batch_size,
        grad_ckpt=args.grad_ckpt,
        train_cond_encoder_only=args.train_cond_encoder_only,
        use_lora=args.use_lora
    )

    trainer.train(
        resumable_with_seed=args.resumable_with_seed,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
