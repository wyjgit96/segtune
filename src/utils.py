import time
import json
import warnings
from pathlib import Path

import torch
import torchaudio
import librosa

warnings.filterwarnings("ignore", message="PySoundFile failed", category=UserWarning)
warnings.filterwarnings("ignore", message=".*__audioread_load.*", category=FutureWarning)


def load_audio(path, target_sr=None, mono=False):
    """
    无感加载 wav/m4a/mp3/flac...
    返回: (waveform[C, T], sample_rate)
    """
    ext = Path(path).suffix.lower()
    if ext in {".wav", ".flac", ".ogg", "mp3"}:
        wav, sr = torchaudio.load(path)
    else:
        y, sr = librosa.load(path, sr=None, mono=False)  # y: [T] or [C, T]
        wav = torch.from_numpy(y) if y.ndim > 1 else torch.from_numpy(y).unsqueeze(0)

    if mono and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def vae_sample(mean, scale):
    stdev = torch.nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = (mean*mean + var - logvar - 1).sum(1).mean()

    return latents, kl


def normalize_audio(y, target_dbfs=0):
    max_amplitude = torch.max(torch.abs(y))

    target_amplitude = 10.0**(target_dbfs / 20.0)
    scale_factor = target_amplitude / max_amplitude

    normalized_audio = y * scale_factor

    return normalized_audio


def set_audio_channels(audio, target_channels):
    if target_channels == 1:
        # Convert to mono
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


class PadCrop(torch.nn.Module):

    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output


def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):

    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = torchaudio.transforms.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)
    if target_length is None:
        target_length = audio.shape[-1]
    audio = PadCrop(target_length, randomize=False)(audio)

    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)

    return audio


############## 歌词预处理 ##############


def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.strip()
    for line in lyrics.split("\n"):
        try:
            time, lyric = line[1:9], line[10:]
            lyric = lyric.strip()
            mins, secs = time.split(":")
            secs = int(mins) * 60 + float(secs)
            lyrics_with_time.append((secs, lyric))
        except:
            continue
    return lyrics_with_time


class CNENTokenizer:

    def __init__(self):
        with open("./src/g2p/g2p/vocab.json", "r", encoding='utf-8') as file:
            self.phone2id: dict = json.load(file)["vocab"]
        self.id2phone = {v: k for (k, v) in self.phone2id.items()}
        from src.g2p.g2p_generation import chn_eng_g2p

        self.tokenizer = chn_eng_g2p

    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x + 1 for x in token]
        return token

    def decode(self, token):
        return "|".join([self.id2phone[x - 1] for x in token])
