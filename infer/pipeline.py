"""
SegTune 端到端推理 Pipeline（TemporalControlDiT 版本）

串联 LRC 时间戳预测（Qwen3 LoRA）与 TemporalControlDiT 扩散模型，
实现从「原始数据文件 → LRC 预测 → 完整歌曲音频」的一键式推理。

两种使用方式（从项目根目录执行）：

  方式一：完整 Pipeline（jsonl 数据文件 → LRC 预测 → 音频生成）
    python infer/pipeline.py \
        --jsonl-path datasets/test/test.jsonl \
        --temporal-config infer/example/temporal_config.json \
        --dit-config-path config/diffrhythm-1b_qwen3.json \
        --output-dir infer/example/output

  方式二：已有 LRC 文件，跳过预测直接生成音频
    python infer/pipeline.py \
        --lrc-path infer/example/input.lrc \
        --temporal-config infer/example/temporal_config.json \
        --dit-config-path config/diffrhythm-1b_qwen3.json

temporal_config.json 格式：
  {
    "global_prompt": "全局风格描述",
    "local_prompts": [
      {"section": [0, 30], "prompt": "verse description"},
      {"section": [30, 60], "prompt": "chorus description"}
    ],
    "default_prompt": "",
    "alpha": 0.2,
    "duration": 95
  }

输入格式说明：
  --jsonl-path  lrc_prediction 模块的标准输入，每行一个 json，含以下字段：
                  lrc_path        带时间戳的 LRC 文件路径（用于提取纯歌词 + 构造 prompt）
                  flamingo_struct.global_analysis     全局歌曲描述
                  flamingo_struct.segment_analyses    段落级描述列表
  --lrc-path    已有带时间戳的 LRC 文件路径（与 infer.py 的 --lrc-path 一致）
"""

import argparse
import json
import os
import sys
import random
import time

import torch
import torchaudio
from einops import rearrange
from huggingface_hub import hf_hub_download
from muq import MuQMuLan

# 确保从项目根目录运行时能找到 src 和 infer 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer.infer_utils import (
    CNENTokenizer,
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    load_checkpoint,
)
from src.lrc_prediction.inference import Inference as LRCInference
from src.lrc_prediction.data_preprocessing import DataPreprocessor
from src.preprocess.qwen3_embedding import Qwen3Embedding
from src.model.temporal_control_dit import TemporalControlDiT, get_temporal_style_prompt
from src.model.cfm import CFM


def prepare_temporal_control_model(
    max_frames,
    device,
    repo_id="ASLP-lab/DiffRhythm-1_2",
    cache_dir="./pretrained",
    model_config=None,
    qwen3_model_path=None,
):
    """准备使用 TemporalControlDiT 的 CFM 模型"""
    if os.path.exists(repo_id):
        dit_ckpt_path = repo_id
    else:
        dit_ckpt_path = hf_hub_download(repo_id=repo_id, filename="cfm_model.pt", cache_dir=cache_dir)

    cond_encoder_type = model_config['cond_encoder']

    cfm = CFM(
        transformer=TemporalControlDiT(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames
    )
    cfm = cfm.to(device)
    cfm = load_checkpoint(cfm, dit_ckpt_path, device=device, use_ema=False)

    tokenizer = CNENTokenizer()

    if cond_encoder_type == 'muq':
        cond_encoder = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=cache_dir)
        cond_encoder = cond_encoder.to(device).eval()
    elif cond_encoder_type == 'qwen3':
        if qwen3_model_path is None:
            raise ValueError("cond_encoder_type='qwen3' 需要指定 --qwen3-model-path")
        cond_encoder = Qwen3Embedding(qwen3_model_path, use_cuda=(device == "cuda"))
        cond_encoder.model = cond_encoder.model.to(device)
    elif cond_encoder_type == 'qwen3_muq':
        if qwen3_model_path is None:
            raise ValueError("cond_encoder_type='qwen3_muq' 需要指定 --qwen3-model-path")
        global_cond_encoder = Qwen3Embedding(qwen3_model_path, use_cuda=(device == "cuda"))
        global_cond_encoder.model = global_cond_encoder.model.to(device)
        local_cond_encoder = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=cache_dir)
        local_cond_encoder = local_cond_encoder.to(device).eval()
        cond_encoder = {"global_encoder": global_cond_encoder, "local_encoder": local_cond_encoder}
    else:
        raise NotImplementedError(f'unknown encoder name {cond_encoder_type}')

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

    required_keys = ["global_prompt", "local_prompts"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件缺少必需的键: {key}")

    if config["local_prompts"]:
        for prompt_info in config["local_prompts"]:
            if "section" not in prompt_info or "prompt" not in prompt_info:
                raise ValueError("local_prompts 中的每个元素必须包含 'section' 和 'prompt' 键")
            if len(prompt_info["section"]) != 2:
                raise ValueError("section 必须是包含 [start_time, end_time] 的列表")

    return config


def run_inference(
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
            output = rearrange(output, "b d n -> d (b n)")
            output = (
                output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            )
            outputs.append(output)

        return outputs


def predict_lrc_from_jsonl(jsonl_path: str, lrc_model_name: str, lrc_lora_dir: str) -> str:
    """
    调用 lrc_prediction 模块，从 jsonl 数据文件预测 LRC。
    取 jsonl 中第一条数据进行预测（pipeline 单曲推理场景）。

    返回：带时间戳的 LRC 字符串
    """
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_all_data(jsonl_path)
    if not processed_data:
        raise ValueError(f"jsonl 文件中没有有效数据: {jsonl_path}")

    predictor = LRCInference(model_name=lrc_model_name, lora_dir=lrc_lora_dir)
    predictor.load_model()

    song_data = processed_data[0]
    lyrics_lines = song_data.get('lyrics_lines', [])
    lrc_text = predictor.predict_lyrics_duration(song_data, lyrics_lines)

    return lrc_text


def main():
    parser = argparse.ArgumentParser(
        description="SegTune Pipeline：LRC 预测 → TemporalControlDiT 音频生成"
    )

    # ── 输入：二选一 ──────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--jsonl-path", type=str,
        help="lrc_prediction 标准输入：jsonl 数据文件路径（含 lrc_path 和歌曲描述）"
    )
    input_group.add_argument(
        "--lrc-path", type=str,
        help="已有带时间戳的 LRC 文件路径，跳过预测直接生成音频"
    )

    # ── 时间控制配置 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--temporal-config", type=str, required=True,
        help="JSON config file for temporal control (global/local prompts with time segments)"
    )

    # ── LRC 预测模型（仅 --jsonl-path 时生效） ────────────────────────────────
    parser.add_argument("--lrc-model-name", type=str,
                        default="Qwen/Qwen3-4B-Base",
                        help="Qwen3 base model name or local path")
    parser.add_argument("--lrc-lora-dir",   type=str, default=None,
                        help="LoRA 权重目录路径（如 exps/train/fine_tuned_model/checkpoint-XXXX）；不指定则用基模 zero-shot")
    parser.add_argument("--lrc-save-path",  type=str, default=None,
                        help="将预测的 LRC 保存到此路径（可选）")

    # ── DiT 模型参数 ─────────────────────────────────────────────────────────
    parser.add_argument("--repo-id",          type=str, default="ASLP-lab/DiffRhythm-1_2")
    parser.add_argument("--dit-config-path",  type=str, required=True,
                        help="模型配置 JSON 文件路径（如 config/diffrhythm-1b_qwen3.json）")
    parser.add_argument("--qwen3-model-path", type=str, default=None,
                        help="Qwen3-Embedding 模型路径（cond_encoder 为 qwen3 或 qwen3_muq 时必填）")
    parser.add_argument("--cache-dir",        type=str, default="./pretrained",
                        help="HuggingFace 模型缓存目录")
    parser.add_argument("--audio-length",     type=int, default=95,
                        help="生成音频时长（秒）")
    parser.add_argument("--output-dir",       type=str, default="infer/example/output")
    parser.add_argument("--chunked",          action="store_true")
    parser.add_argument("--batch-infer-num",  type=int, default=1)
    parser.add_argument("--merge-strategy",   type=str, default="concat",
                        help="全局/局部 style prompt 的合并策略")
    parser.add_argument("--cfg",              type=float, default=4.0,
                        help="CFG strength（主提示引导强度）")
    parser.add_argument("--cfg2",             type=float, default=4.0,
                        help="CFG strength2（时间控制引导强度）")

    # ── 编辑模式 ─────────────────────────────────────────────────────────────
    parser.add_argument("--edit",          action="store_true")
    parser.add_argument("--ref-song",      type=str, default=None)
    parser.add_argument("--edit-segments", type=str, default=None,
                        help="Time segments to edit (in seconds). Format: `[[start1,end1],...]`. "
                             "Use `-1` for audio start/end (e.g., `[[-1,25], [50.0,-1]]`).")

    args = parser.parse_args()

    if args.edit:
        assert (args.ref_song and args.edit_segments), \
            "reference song and edit segments should be provided for editing"

    # ── 设备 & max_frames ─────────────────────────────────────────────────────
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    max_frames = int(args.audio_length * 44100 / 2048)

    # ── Step 1: 获取 LRC ──────────────────────────────────────────────────────
    if args.lrc_path:
        with open(args.lrc_path, "r", encoding="utf-8") as f:
            lrc = f.read()
        print(f"[Pipeline] 使用已有 LRC 文件: {args.lrc_path}")
    else:
        print("[Pipeline] Step 1: 预测 LRC 时间戳...")
        t0 = time.time()
        lrc = predict_lrc_from_jsonl(args.jsonl_path, args.lrc_model_name, args.lrc_lora_dir)
        print(f"[Pipeline] LRC 预测完成，耗时 {time.time() - t0:.2f}s")

        if args.lrc_save_path:
            os.makedirs(os.path.dirname(args.lrc_save_path) or ".", exist_ok=True)
            with open(args.lrc_save_path, "w", encoding="utf-8") as f:
                f.write(lrc)
            print(f"[Pipeline] LRC 已保存到: {args.lrc_save_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Step 2: 加载 TemporalControlDiT 模型 ─────────────────────────────────
    print("[Pipeline] Step 2: 加载 TemporalControlDiT 模型...")
    with open(args.dit_config_path) as f:
        model_config = json.load(f)

    cfm, tokenizer, cond_encoder, vae = prepare_temporal_control_model(
        max_frames,
        device,
        repo_id=args.repo_id,
        cache_dir=args.cache_dir,
        model_config=model_config,
        qwen3_model_path=args.qwen3_model_path,
    )

    lrc_prompt, start_time = get_lrc_token(max_frames, lrc, tokenizer, device)

    # ── Step 3: 构建时间控制风格提示 ─────────────────────────────────────────
    print("[Pipeline] Step 3: 构建时间控制风格提示...")
    config = load_temporal_control_config(args.temporal_config)
    audio_duration = config.get("duration", args.audio_length)

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

    # ── Step 4: 生成音频 ──────────────────────────────────────────────────────
    print("[Pipeline] Step 4: 生成音频...")
    s_t = time.time()
    generated_songs = run_inference(
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
    print(f"[Pipeline] 音频生成耗时 {time.time() - s_t:.2f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    max_samples = int(audio_duration * 44100)
    for index, generated_song in enumerate(generated_songs):
        if generated_song.shape[1] > max_samples:
            generated_song = generated_song[:, :max_samples]
        output_path = os.path.join(args.output_dir, f"output_{index}.wav")
        torchaudio.save(output_path, generated_song, sample_rate=44100)
        print(f"[Pipeline] ✅ 完成！输出文件：{output_path}")


if __name__ == "__main__":
    main()
