from __future__ import annotations
from typing import Union

import torch
from torch import nn

from src.model.dit import DiT, InputEmbedding, TextEmbedding
from src.model.modules import ConvPositionEmbedding, _prepare_decoder_attention_mask


class TemporalControlInputEmbedding(InputEmbedding):
    """扩展 InputEmbedding 类以处理时序控制的 style embedding。
    """

    def __init__(self, mel_dim, text_dim, out_dim, cond_dim, time_embed_dim=512, cond_input_dim=None):
        super().__init__(mel_dim, text_dim, out_dim, cond_dim, time_embed_dim)
        if cond_input_dim:
            self.cond_proj = nn.Sequential(
                nn.Conv1d(cond_input_dim, cond_input_dim, 1),
                nn.SiLU(),
                nn.Conv1d(cond_input_dim, cond_input_dim, 1),
                nn.SiLU(),
                nn.Conv1d(cond_input_dim, cond_dim, 1),
            )
        else:
            self.cond_proj = nn.Identity()

    def forward(
        self,
        x: float["b n d"],  # noqa: F722
        cond: float["b n d"],  # noqa: F722
        text_embed: float["b n d"],  # noqa: F722
        style_emb,
        time_emb,
        drop_audio_cond=False
    ):
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        time_emb = time_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        if style_emb.ndim == 2:
            # 对 global prompt, 直接广播
            style_emb = style_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        style_emb = self.cond_proj(style_emb.transpose(-1, -2)).transpose(-1, -2)

        x = self.proj(torch.cat((x, cond, text_embed, style_emb, time_emb), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class TemporalControlDiT(DiT):
    """扩展DiT模型以支持时序控制的样式嵌入。
    
    这个模型与原始DiT的主要区别在于:
    - 原始DiT: style_emb (style_prompt) 维度为 [b, d]
    - TemporalControlDiT: style_emb 维度为 [b, n, d]
    
    这要求对InputEmbedding进行修改, 以适应不同维度的style_emb输入。
    """

    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        max_frames=2048,
        time_embed_dim=512,
        cond_dim=512,
        cond_input_dim=1024,
        **kwargs,
    ):
        # 调用DiT的初始化，但不使用其input_embed
        super().__init__(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            ff_mult=ff_mult,
            mel_dim=mel_dim,
            text_num_embeds=text_num_embeds,
            text_dim=text_dim,
            conv_layers=conv_layers,
            long_skip_connection=long_skip_connection,
            max_frames=max_frames,
            time_embed_dim=time_embed_dim,
            cond_dim=cond_dim,
        )
        # 替换为我们自定义的TemporalControlInputEmbedding
        if text_dim is None:
            text_dim = mel_dim
        # TODO：cond_input_dim 参数目前是写死的，后期需要支持通过配置文件修改
        self.input_embed = TemporalControlInputEmbedding(
            mel_dim,
            text_dim,
            dim,
            cond_dim=cond_dim,
            time_embed_dim=time_embed_dim,
            cond_input_dim=cond_input_dim,
        )


@torch.no_grad()
def get_temporal_style_prompt(
    model,
    n_frames,
    global_prompt: str = "",
    local_prompts: list = None,
    default_prompt: Union[str] = None,
    alpha=0.5,
    sampling_rate=44100,
    downsample_rate=2048,
    merge_strategy='concat',
    local_dim=512,
):
    """
    生成支持时间控制的风格嵌入向量。
    
    Args:
        model: 风格嵌入编码器
        n_frames: latent 的时间维长度
        global_prompt: 全局风格提示 (str)，控制整首歌的风格
        local_prompts: 局部风格提示列表，每个元素为字典 {"section": [start_time, end_time], "prompt": ...}
        default_prompt: 默认提示 (str)，用于没有被局部提示覆盖的区域
        alpha: 融合权重，mix_embedding = alpha * global_embedding + (1 - alpha) * local_embedding
        sampling_rate: 音频采样率，默认 44100
        downsample_rate: 下采样率，默认 2048
        merge_method: 全局-局部特征融合方式，可选类型包括 
            - "concat": 两类特征在通道维度拼接；
            - "mix": 两类特征加权平均；
    
    Returns:
        torch.Tensor: 维度为 (1, n_frames, d) 的时间控制风格嵌入
    """
    if isinstance(model, dict):
        global_model = model["global_encoder"]
        local_model = model["local_encoder"]
        device = global_model.device
    else:
        global_model = local_model = model
        device = model.device

    # 获取全局风格嵌入 (1, d)
    global_embedding = global_model(texts=global_prompt).half()  # (1, d)
    global_embedding_expanded = global_embedding.unsqueeze(1).repeat(1, n_frames, 1)  # (1, n_frames, d)

    # 处理局部提示
    if local_prompts is not None and len(local_prompts) > 0 and merge_strategy != 'global_only':
        # 初始化局部嵌入矩阵 (n_frames, d)
        local_embedding_matrix = torch.zeros(n_frames, local_dim, device=device, dtype=torch.float16)

        for local_prompt_info in local_prompts:
            section = local_prompt_info["section"]
            prompt = local_prompt_info["prompt"]

            start_time, end_time = section

            # 将时间转换为帧索引
            start_frame = int(start_time * sampling_rate / downsample_rate)
            end_frame = int(end_time * sampling_rate / downsample_rate)

            start_frame = max(0, min(start_frame, n_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, n_frames))

            # 获取局部嵌入 (1, d), 填充到对应区域
            local_embedding = local_model(texts=prompt).half()
            assert local_embedding.shape[-1] == local_dim, \
                f"dim of local embedding is {local_embedding.shape[-1]}, but it's expected to be {local_dim},"
            local_embedding_matrix[start_frame:end_frame] = local_embedding.squeeze(0)
            # 扩展维度以匹配 (1, n_frames, d) 格式
            local_embedding_tensor = local_embedding_matrix.unsqueeze(0)  # (1, n_frames, d)

    if merge_strategy == 'mix':
        # 融合全局和局部嵌入
        out_embedding = alpha*global_embedding_expanded + (1-alpha) * local_embedding_tensor
    elif merge_strategy == 'concat':
        out_embedding = torch.cat([global_embedding_expanded, local_embedding_tensor], dim=-1)
    elif merge_strategy == 'global_only':
        out_embedding = global_embedding_expanded
    else:
        raise NotImplementedError(f"Unknown merge strategy: {merge_strategy}")
    return out_embedding
