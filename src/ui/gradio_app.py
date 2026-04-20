import pdb
import gradio as gr
import os
import sys
import time
import torch
import torchaudio
from einops import rearrange
import json
from huggingface_hub import hf_hub_download

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'infer'))
os.chdir(project_root)

from src.lrc_gen.composer import Composer
from muq import MuQMuLan
from src.model import DiT, CFM
from infer import inference
from infer_utils import (
    prepare_model,
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    CNENTokenizer,
    load_checkpoint,
)


class Performer:
    """演奏者类，负责生成音乐"""

    def __init__(self, repo_id="ASLP-lab/DiffRhythm-1_2", max_length=95):
        """初始化演奏者
        
        Args:
            repo_id: 模型仓库 ID
        """
        self.repo_id = repo_id
        self.device = self._get_device()
        self.cache_dir = os.path.join(project_root, "src", "ui", ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cfm, self.tokenizer, self.muq, self.vae = prepare_model(
            int((2048/95) * max_length),
            self.device,
            repo_id=self.repo_id,
        )
        print(f"演奏者初始化完成，使用设备: {self.device}")

    def _get_device(self):
        """获取计算设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def generate_music(self, lrc_content: str, tags: str, duration: int) -> str:
        """生成音乐
        
        Args:
            lrc_content: LRC 歌词内容
            tags: 音乐标签
            duration: 音频总时长
            
        Returns:
            str: 生成的音频文件路径或错误信息
        """
        try:
            # 检查输入
            if not lrc_content or not lrc_content.strip():
                return "错误：请先生成 LRC 歌词"
            if duration <= 0:
                return "错误：请先生成有效的音频时长"

            # 设置音频长度
            audio_length = min(max(duration, 30), 285)  # 限制在 30-285 秒之间
            frame_length = int((2048/95) * audio_length)
            self.cfm.max_frames = frame_length

            # 每次都重新初始化模型
            print(f"正在初始化模型，音频长度: {audio_length}秒，max_frames: {frame_length}")

            # 使用时间戳创建唯一文件名
            timestamp = int(time.time())
            temp_lrc_path = os.path.join(self.cache_dir, f"temp_{timestamp}.lrc")

            # 写入 LRC 内容
            with open(temp_lrc_path, "w", encoding='utf-8') as f:
                f.write(lrc_content)

            # 获取 LRC token
            lrc_prompt, start_time = get_lrc_token(frame_length, lrc_content, self.tokenizer, self.device)

            # 获取风格提示（使用文本提示）
            style_prompt = get_style_prompt(self.muq, prompt=tags)
            negative_style_prompt = get_negative_style_prompt(self.device)

            # 获取参考潜在向量（无编辑模式）
            latent_prompt, pred_frames = get_reference_latent(self.device, frame_length, False, None, None, self.vae)

            # 执行推理
            print(f"开始生成音乐，时长: {audio_length} 秒")
            s_t = time.time()
            generated_songs = inference(
                cfm_model=self.cfm,
                vae_model=self.vae,
                cond=latent_prompt,
                text=lrc_prompt,
                duration=frame_length,
                style_prompt=style_prompt,
                negative_style_prompt=negative_style_prompt,
                start_time=start_time,
                pred_frames=pred_frames,
                chunked=True,
                batch_infer_num=1
            )
            e_t = time.time() - s_t
            print(f"音乐生成完成，耗时 {e_t:.2f} 秒")

            # 保存生成的音频
            generated_song = generated_songs[0]
            output_path = os.path.join(self.cache_dir, f"generated_music_{timestamp}.wav")
            torchaudio.save(output_path, generated_song, sample_rate=44100)
            return output_path

        except Exception as e:
            error_msg = f"音乐生成失败: {str(e)}"
            print(error_msg)
            return error_msg


composer = Composer(os.path.join(project_root, "config", "gpt_config.json"))
performer = Performer()


def generate_lrc(music_name: str, tags: str, lyrics: str) -> tuple[str, int]:
    """生成 LRC 格式歌词
    
    Args:
        music_name: 音乐名称
        tags: 音乐标签
        lyrics: 歌词内容
        
    Returns:
        tuple: (处理后的LRC内容, 音频总时长)
    """
    try:
        # 构建歌曲描述
        song_description = f"音乐名: {music_name}\n标签: {tags}"

        # 生成 LRC 内容
        lrc_content = composer.generate_lrc(lyrics, song_description)

        # 获取音频总时长
        duration = composer.get_song_duration(lrc_content)

        # 去除结构化标签
        processed_lrc = composer.remove_structural_tags(lrc_content)

        return processed_lrc, duration

    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        return error_msg, 0


def create_gradio_interface():
    """创建 Gradio 界面"""
    with gr.Blocks(title="Kling-musci", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Kling-music")
        gr.Markdown("根据音乐信息和歌词生成带时间轴的 LRC 格式歌词，并生成对应的音乐")

        with gr.Row():
            # 左侧输入栏
            with gr.Column(scale=1):
                gr.Markdown("## 输入信息")

                music_name_input = gr.Textbox(label="音乐名", placeholder="请输入音乐名称", lines=1)
                tags_input = gr.Textbox(
                    label="Tags",
                    placeholder="请输入音乐标签（英文），如: pop(曲风), piano(乐器), happy(情感), young woman(歌者音色) ...",
                    lines=1
                )

                lyrics_input = gr.Textbox(label="歌词", placeholder="请输入完整歌词...", lines=15, max_lines=20)

                generate_lrc_btn = gr.Button("生成 LRC 格式", variant="primary", size="lg")

            # 右侧输出栏
            with gr.Column(scale=1):
                gr.Markdown("## 生成结果")

                lrc_output = gr.Textbox(
                    label="LRC 歌词", placeholder="生成的 LRC 格式歌词将显示在这里...", lines=12, max_lines=15, interactive=True
                )

                duration_output = gr.Number(label="音频总时长（秒）", placeholder="音频总时长将显示在这里", interactive=True)

                generate_music_btn = gr.Button("生成音乐", variant="secondary", size="lg")

                audio_output = gr.Audio(label="生成的音乐", type="filepath", interactive=False)

        # 绑定生成 LRC 按钮事件
        generate_lrc_btn.click(
            fn=generate_lrc,
            inputs=[music_name_input, tags_input, lyrics_input],
            outputs=[lrc_output, duration_output],
        )

        # 绑定生成音乐按钮事件
        def generate_music_wrapper(music_name, tags, lrc_content, duration):
            return performer.generate_music(lrc_content, tags, duration)

        generate_music_btn.click(
            fn=generate_music_wrapper,
            inputs=[music_name_input, tags_input, lrc_output, duration_output],
            outputs=[audio_output]
        )

        # 添加使用说明
        with gr.Accordion("使用说明", open=True):
            gr.Markdown(
                """
            ### 使用步骤：
            1. **音乐名**：输入歌曲名称
            2. **Tags**：输入音乐风格标签，用逗号分隔
            3. **歌词**：输入完整的歌词内容
            4. 点击 **"生成 LRC 格式"** 按钮
            5. 右侧将显示生成的 LRC 格式歌词和音频总时长
            6. 点击 **"生成音乐"** 按钮，根据 LRC 和音乐描述生成音乐
            7. 生成的音乐将在音频播放器中显示
            
            注意： 当前版本不支持 95s 以上歌曲生成
            """
            )

    app.launch(server_name="0.0.0.0", server_port=8890, share=True)


if __name__ == "__main__":
    # 创建并启动界面
    create_gradio_interface()
