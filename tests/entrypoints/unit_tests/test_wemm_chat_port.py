# SPDX-License-Identifier: Apache-2.0
"""WeMM dispatch must not change the existing model media paths."""

from unittest.mock import Mock

import pytest

from vllm.entrypoints.chat_utils import (
    MODALITY_PLACEHOLDERS_MAP,
    _parse_chat_message_content_part,
)


def parse(part, parser, structured=False):
    return _parse_chat_message_content_part(
        part, parser, wrap_dicts=structured, interleave_strings=True
    )


def test_input_audio_uses_wemm_connector():
    parser = Mock(_wemm_enabled=True)
    parser.mm_placeholder_storage.return_value = {
        MODALITY_PLACEHOLDERS_MAP["audio"]: ["<|audio_start|><|audio_pad|><|audio_end|>"]
    }
    parse({"type": "input_audio", "input_audio": {"format": "wav", "data": "abc"}}, parser)
    parser.parse_wemm_audio.assert_called_once_with("data:audio/wav;base64,abc", None)
    parser.parse_input_audio.assert_not_called()


def test_standard_models_keep_input_audio_and_video_paths():
    parser = Mock(_wemm_enabled=False)
    audio = {"format": "wav", "data": "abc"}
    parse({"type": "input_audio", "input_audio": audio}, parser)
    parse({"type": "video_url", "video_url": {"url": "video", "use_audio": True}}, parser)
    parser.parse_input_audio.assert_called_once_with(audio, None)
    parser.parse_video.assert_called_once_with("video", None)
    parser.parse_audio.assert_not_called()


def test_audio_url_timestamp_and_structured_placeholder():
    parser = Mock(_wemm_enabled=True)
    placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
    parser.mm_placeholder_storage.return_value = {
        MODALITY_PLACEHOLDERS_MAP["audio"]: [placeholder]
    }
    result = parse(
        {"type": "audio_url", "audio_url": {"url": "audio", "timestamp_interval_sec": 2.0}},
        parser, structured=True,
    )
    parser.parse_wemm_audio.assert_called_once_with("audio", None, 2.0)
    assert result == {"type": "text", "text": placeholder}


def test_paired_video_keeps_audio_inside_one_video_item():
    parser = Mock(_wemm_enabled=True)
    parse({
        "type": "video_audio_url",
        "video_url": {"url": "video", "fps": 2},
        "audio_url": {"url": "audio"},
    }, parser)
    parser.parse_wemm_video.assert_called_once_with("video", None, audio_url="audio", fps=2)
    parser.parse_audio.assert_not_called()


def test_custom_video_type_rejected_for_other_models():
    with pytest.raises(ValueError, match="only supported by WeMM"):
        parse({"type": "video_audio_url", "video_url": {"url": "video"}},
              Mock(_wemm_enabled=False))
