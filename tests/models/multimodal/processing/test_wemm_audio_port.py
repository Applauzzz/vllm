# SPDX-License-Identifier: Apache-2.0
import io
from unittest.mock import patch

import numpy as np
import soundfile as sf

from vllm.multimodal.wemm_audio import _decode_audio_to_16k_mono
from vllm.utils.import_utils import PlaceholderModule


def test_wav_resampling_without_librosa_or_ffmpeg():
    samples = np.sin(np.arange(24000) * 2 * np.pi * 440 / 24000).astype(np.float32) * 0.1
    buffer = io.BytesIO()
    sf.write(buffer, samples, 24000, format="WAV", subtype="PCM_16")
    with (
        patch("vllm.multimodal.wemm_audio.TorchcodecAudioDecoder", None),
        patch("vllm.multimodal.wemm_audio.librosa", PlaceholderModule("librosa")),
        patch("subprocess.run", side_effect=AssertionError("must not need ffmpeg")),
    ):
        result = _decode_audio_to_16k_mono(buffer.getvalue())[0]
    assert result.shape == (16000,) and result.dtype == np.float32
    assert 0.065 < np.sqrt(np.mean(result ** 2)) < 0.075
