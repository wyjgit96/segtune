from importlib.resources import files

from src.model import CFM, DiT, Trainer

from prefigure.prefigure import get_all_args
import json
import os

os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"


def main():
    args = get_all_args("config/default.ini")

    with open(args.model_config) as f:
        model_config = json.load(f)

    if model_config["model_type"] == "diffrhythm":
        wandb_resume_id = None
        model_cls = DiT

    model = CFM(
        transformer=model_cls(**model_config["model"], max_frames=args.max_frames),
        num_channels=model_config["model"]['mel_dim'],
        audio_drop_prob=args.audio_drop_prob,
        cond_drop_prob=args.cond_drop_prob,
        style_drop_prob=args.style_drop_prob,
        lrc_drop_prob=args.lrc_drop_prob,
        max_frames=args.max_frames
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    trainer = Trainer(
        model,
        args,  # passed as 'arguments'
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        checkpoint_path=f"ckpts/{args.exp_name}",
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        wandb_project="diffrhythm-test",
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=args.last_per_steps,
        bnb_optimizer=False,
        reset_lr=args.reset_lr,
        batch_size=args.batch_size,
        grad_ckpt=args.grad_ckpt
    )

    trainer.train(
        resumable_with_seed=args.resumable_with_seed,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
