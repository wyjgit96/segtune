import pdb
import argparse
import os
import time
import random
import json

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from muq import MuQMuLan
from einops import rearrange

print("Current working directory:", os.getcwd())

from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
    CNENTokenizer,
    load_checkpoint,
)
from src.preprocess.qwen3_embedding import Qwen3Embedding
from src.model.temporal_control_dit import TemporalControlDiT, get_temporal_style_prompt
from src.model.cfm import CFM


def prepare_temporal_control_model(
    max_frames,
    device,
    repo_id="ASLP-lab/DiffRhythm-1_2",
    cache_dir="./pretrained",
    model_config=None,
):
    """准备使用TemporalControlDiT的CFM模型"""
    if os.path.exists(repo_id):
        dit_ckpt_path = repo_id
    else:
        dit_ckpt_path = hf_hub_download(repo_id=repo_id, filename="cfm_model.pt", cache_dir=cache_dir)

    cond_encoder_type = model_config['cond_encoder']

    # 使用TemporalControlDiT替代DiT
    dit_model_cls = TemporalControlDiT
    cfm = CFM(
        transformer=dit_model_cls(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames
    )
    cfm = cfm.to(device)
    cfm = load_checkpoint(cfm, dit_ckpt_path, device=device, use_ema=False)

    # 准备tokenizer
    tokenizer = CNENTokenizer()

    # 准备muq
    if cond_encoder_type == 'muq':
        cond_encoder = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=cache_dir)
        cond_encoder = cond_encoder.to(device).eval()
    elif cond_encoder_type == 'qwen3':
        model_path = "/ytech_milm_disk2/cai_pengfei/music_generation/DiffRhythm/pretrained/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
        cond_encoder = Qwen3Embedding(model_path, use_cuda=True)
        cond_encoder.model = cond_encoder.model.cuda(device)
    elif cond_encoder_type == 'qwen3_muq':
        # global encoder
        model_path = "/ytech_milm_disk2/cai_pengfei/music_generation/DiffRhythm/pretrained/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
        global_cond_encoder = Qwen3Embedding(model_path, use_cuda=True)
        global_cond_encoder.model = global_cond_encoder.model.cuda(device)
        # local encoder
        local_cond_encoder = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=cache_dir)
        local_cond_encoder = local_cond_encoder.to(device).eval()
        # package
        cond_encoder = {"global_encoder": global_cond_encoder, "local_encoder": local_cond_encoder}
    else:
        raise NotImplementedError(f'unknown encode name {cond_encoder_type}')

    # 准备vae
    vae_ckpt_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt",
        cache_dir=cache_dir,
    )
    vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)

    return cfm, tokenizer, cond_encoder, vae


def load_temporal_control_config(config_path):
    """加载时间控制配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 验证配置格式
    required_keys = ["global_prompt", "local_prompts"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件缺少必需的键: {key}")

    # 验证local_prompts格式
    if config["local_prompts"]:
        for prompt_info in config["local_prompts"]:
            if "section" not in prompt_info or "prompt" not in prompt_info:
                raise ValueError("local_prompts中的每个元素必须包含'section'和'prompt'键")
            if len(prompt_info["section"]) != 2:
                raise ValueError("section必须是包含[start_time, end_time]的列表")

    return config


def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    pred_frames,
    batch_infer_num,
    chunked=False,
    steps=32,
    cfg_strength=4.0,
    cfg_strength2=4.0,
):
    with torch.inference_mode():
        latents, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=steps,
            cfg_strength=cfg_strength,
            cfg_strength2=cfg_strength2,
            start_time=start_time,
            latent_pred_segments=pred_frames,
            batch_infer_num=batch_infer_num,
        )

        outputs = []
        for latent in latents:
            latent = latent.to(torch.float32)
            latent = latent.transpose(1, 2)  # [b d t]

            output = decode_audio(latent, vae_model, chunked=chunked)

            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")
            # Peak normalize, clip, convert to int16, and save to file
            output = (
                output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            )
            outputs.append(output)

        return outputs


if __name__ == "__main__":
    # pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lrc-path",
        type=str,
        help="lyrics of target song",
    )  # lyrics of target song
    parser.add_argument(
        "--temporal-config",
        type=str,
        help="JSON config file for temporal control (global and local prompts)",
        required=False,
    )
    parser.add_argument(
        "--ref-prompt",
        type=str,
        help="reference prompt as style prompt for target song (fallback if no temporal config)",
        required=False,
    )  # reference prompt as style prompt for target song
    parser.add_argument(
        "--ref-audio-path",
        type=str,
        help="reference audio as style prompt for target song (fallback if no temporal config)",
        required=False,
    )  # reference audio as style prompt for target song
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="whether to use chunked decoding",
    )  # whether to use chunked decoding
    parser.add_argument(
        "--audio-length",
        type=int,
        default=95,
        # choices=[95, 285],
        help="length of generated song",
    )  # length of target song
    parser.add_argument("--repo-id", type=str, default="ASLP-lab/DiffRhythm-base", help="target model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="infer/example/output",
        help="output directory fo generated song",
    )  # output directory of target song
    parser.add_argument(
        "--edit",
        action="store_true",
        help="whether to open edit mode",
    )  # edit flag
    parser.add_argument(
        "--ref-song",
        type=str,
        required=False,
        help="reference prompt as latent prompt for editing",
    )  # reference prompt as latent prompt for editing
    parser.add_argument(
        "--edit-segments",
        type=str,
        required=False,
        help="Time segments to edit (in seconds). Format: `[[start1,end1],...]`. "
        "Use `-1` for audio start/end (e.g., `[[-1,25], [50.0,-1]]`)."
    )  # edit segments of target song
    parser.add_argument(
        "--batch-infer-num",
        type=int,
        default=1,
        required=False,
        help="number of songs per batch",
    )  # number of songs per batch
    parser.add_argument(
        "--merge-strategy",
        type=str,
        default='concat',
        required=False,
    )

    parser.add_argument(
        "--dit-config-path",
        type=str,
        default='"./config/diffrhythm-1b_qwen3.json"',
        required=False,
    )
    parser.add_argument(
        "--cfg",
        type=int,
        default=3,
        required=False,
    )
    parser.add_argument(
        "--cfg2",
        type=int,
        default=3,
        required=False,
    )
    args = parser.parse_args()

    # 检查参数有效性
    if args.temporal_config:
        print("使用时间控制模式")
    else:
        assert (
            args.ref_prompt or args.ref_audio_path
        ), "either temporal_config, ref_prompt or ref_audio_path should be provided"
        assert not (args.ref_prompt and args.ref_audio_path), "only one of them should be provided"

    if args.edit:
        assert (args.ref_song and args.edit_segments), "reference song and edit segments should be provided for editing"

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    audio_length = args.audio_length
    max_frames = int(audio_length * 44100 / 2048)  # TODO: 暂时写死了采样率以及下采样系数，之后注意修改

    with open(args.dit_config_path) as f:
        model_config = json.load(f)

    cfm, tokenizer, cond_encoder, vae = prepare_temporal_control_model(
        max_frames,
        device,
        repo_id=args.repo_id,
        model_config=model_config,
    )

    if args.lrc_path:
        with open(args.lrc_path, "r", encoding='utf-8') as f:
            lrc = f.read()
    else:
        lrc = ""
    lrc_prompt, start_time = get_lrc_token(max_frames, lrc, tokenizer, device)

    # 获取风格提示
    config = load_temporal_control_config(args.temporal_config)
    audio_duration = config['duration']
    style_prompt = get_temporal_style_prompt(
        model=cond_encoder,
        n_frames=max_frames,
        global_prompt=config["global_prompt"],
        local_prompts=config["local_prompts"],
        default_prompt=config.get("default_prompt", ""),
        alpha=config.get("alpha", 0.2),
        merge_strategy=args.merge_strategy,
        local_dim=model_config["model"]["cond_local_dim"],
    )

    negative_style_prompt = get_negative_style_prompt(
        device,
        style_prompt,
        cond_encoder,
        merge_strategy=args.merge_strategy,
    )

    latent_prompt, pred_frames = get_reference_latent(
        device, max_frames, args.edit, args.edit_segments, args.ref_song, vae
    )

    print("开始推理...")
    s_t = time.time()
    generated_songs = inference(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=max_frames,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        pred_frames=pred_frames,
        chunked=args.chunked,
        batch_infer_num=args.batch_infer_num,
        cfg_strength=args.cfg,
        cfg_strength2=args.cfg2,
    )
    e_t = time.time() - s_t
    print(f"推理耗时 {e_t:.2f} 秒")

    generated_song = random.sample(generated_songs, 1)[0]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # output_path = os.path.join(output_dir, "temporal_control_output.wav")
    # torchaudio.save(output_path, generated_song, sample_rate=44100)

    # inference gives #(args.batch_infer_num) songs at once, you can save all/ any one of them:
    for index, generated_song in enumerate(generated_songs):
        # 计算需要保留的样本数（44100是采样率）
        max_samples = int(audio_duration * 44100)
        # 截断音频到指定时长
        if generated_song.shape[1] > max_samples:
            generated_song = generated_song[:, :max_samples]

        output_path = os.path.join(output_dir, "temporal_control_song_sample_" + str(index) + ".wav")
        torchaudio.save(output_path, generated_song, sample_rate=44100)

    print(f"音频已保存到: {output_path}")
