# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import random

# We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
MODEL_PATH = "/src/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH


import shutil

from tempfile import TemporaryDirectory
from pathlib import Path
from distutils.dir_util import copy_tree
from typing import Optional
from cog import BasePredictor, Input, Path
import torch
import datetime

# Model specific imports
import torchaudio
import subprocess
import typing as tp
import numpy as np

from audiocraft.models import MusicGen
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
    HF_MODEL_CHECKPOINTS_MAP,
)
from audiocraft.data.audio import audio_write


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.small_model = self._load_model(
            model_path=MODEL_PATH,
            cls=MusicGen,
            model_id="facebook/musicgen-small",
        )


    def _load_model(
        self,
        model_path: str,
        cls: Optional[any] = None,
        load_args: Optional[dict] = {},
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> MusicGen:

        if device is None:
            device = self.device

        name = next(
            (key for key, val in HF_MODEL_CHECKPOINTS_MAP.items() if val == model_id),
            None,
        )
        compression_model = load_compression_model(
            name, device=device, cache_dir=model_path
        )
        lm = load_lm_model(name, device=device, cache_dir=model_path)

        return MusicGen(name, compression_model, lm)

    def predict(
        self,
        prompt: str = Input(
            description="A description of the music you want to generate.", default=None
        ),
        duration: int = Input(
            description="Duration of the generated audio in seconds.", default=8, le=30
        ),
        normalization_strategy: str = Input(
            description="Strategy for normalizing audio.",
            default="loudness",
            choices=["loudness", "clip", "peak", "rms"],
        ),
        top_k: int = Input(
            description="Reduces sampling to the k most likely tokens.", default=250
        ),
        top_p: float = Input(
            description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
            default=0.0,
        ),
        temperature: float = Input(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
            default=1.0,
        ),
        classifier_free_guidance: int = Input(
            description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
            default=3,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
        seed: int = Input(
            description="Seed for random number generator. If None or -1, a random seed will be used.",
            default=None,
        ),
    ) -> Path:

        if prompt is None:
            raise ValueError("Must provide either prompt or input_audio")

        model = self.small_model

        set_generation_params = lambda duration: model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )

        set_generation_params(duration)
        wav = model.generate([prompt], progress=True)

        audio_write(
            "out",
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
        )
        wav_path = "out.wav"

        if output_format == "mp3":
            mp3_path = "out.mp3"
            subprocess.call(["ffmpeg", "-i", wav_path, mp3_path])
            os.remove(wav_path)
            path = mp3_path
        else:
            path = wav_path

        return Path(path)

# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
