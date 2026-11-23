# SPDX-License-Identifier: Apache-2.0
"""Focused contracts for the WeMM overlay on the local processor API."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.wemm_omni import (
    WeMMDataParser,
    WeMMDummyInputsBuilder,
    WeMMMultiModalProcessor,
    _split_audio_embeds_by_items,
    merge_multimodal_embeddings,
)
from vllm.model_executor.models.wemm_processor import compute_audio_tokens_per_segment
from vllm.multimodal.media.wemm import wemm_audio_item
from vllm.multimodal.processing import BaseMultiModalProcessor


def test_dense_and_moe_architecture_resolution():
    for model_type, expected in (
        ("qwen3_5_text", "WeMMVideoForConditionalGeneration"),
        ("qwen3_5_moe_text", "WeMMVideoMoeForConditionalGeneration"),
    ):
        config = SimpleNamespace(
            hf_config=SimpleNamespace(text_config=SimpleNamespace(model_type=model_type))
        )
        assert ModelRegistry._normalize_arch(
            "WeMMVideoForConditionalGeneration", config
        ) == expected


def test_audio_policy_survives_data_parser():
    audio = (np.zeros(16000, dtype=np.float32), 16000)
    wrapped, first_uuid = wemm_audio_item(audio, "same", 1.0)
    _, second_uuid = wemm_audio_item(audio, "same", 2.0)
    assert first_uuid != second_uuid
    items = WeMMDataParser(2, target_sr=16000).parse_mm_data({"audio": [wrapped]})
    assert items["audio"].get_audio_length(0) == 16000
    assert items["audio"].get_processor_data()["audios"][0]["timestamp_interval_sec"] == 1.0


def test_local_processor_entrypoint_calls_wemm():
    cache = object()
    info = SimpleNamespace(ctx=SimpleNamespace(tokenizer=object()),
                           get_data_parser=lambda: object())
    constructed = WeMMMultiModalProcessor(info, object(), cache=cache)
    assert constructed.cache is cache
    processor = object.__new__(WeMMMultiModalProcessor)
    processor.dummy_inputs = SimpleNamespace(get_dummy_text=lambda counts: "audio")
    processor.info = SimpleNamespace(
        ctx=SimpleNamespace(_postprocess_output=lambda data: data)
    )
    items = SimpleNamespace(
        get_all_counts=lambda: {"audio": 1},
        select=lambda keys: items,
    )
    with (
        patch.object(processor, "_get_hf_mm_data", return_value=({"audios": [1]}, {})),
        patch.object(
            processor, "_call_wemm_hf_processor",
            return_value=BatchFeature({"audio_feature": torch.ones(1, 128, 3000)}),
        ) as call,
    ):
        result = processor._apply_hf_processor_main(items, {})
    call.assert_called_once_with("audio", {"audios": [1]}, {}, {})
    assert result["audio_feature"].shape == (1, 128, 3000)


def test_local_prompt_update_signature():
    processor = object.__new__(WeMMMultiModalProcessor)
    processor._full_bypass = False
    processor.info = SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(
            video_token_id=3, vision_start_token_id=4, vision_end_token_id=5
        )
    )
    with patch.object(
        BaseMultiModalProcessor, "_maybe_apply_prompt_updates",
        return_value=([1, 2], {}),
    ) as call:
        assert processor._maybe_apply_prompt_updates({}, [1, 2], {}, {}) == ([1, 2], {})
    call.assert_called_once_with({}, [1, 2], {}, {})


def test_audio_embedding_count_and_merge():
    info = SimpleNamespace(
        get_image_size_with_most_features=lambda: (28, 28),
        get_num_frames_with_most_features=lambda seq_len, counts: 2,
        get_video_processor=lambda: object(),
        _get_vision_info=lambda **kwargs: (SimpleNamespace(width=28, height=28), None),
    )
    dummy = WeMMDummyInputsBuilder(info).get_dummy_mm_data(2048, {"audio": 2})
    assert len(dummy["audio"]) == 2 and len(dummy["audio"][0]) == 30 * 16000
    assert compute_audio_tokens_per_segment(0.5) == 60
    encoded = torch.arange(240, dtype=torch.float32).reshape(120, 2)
    parts = _split_audio_embeds_by_items(encoded, [1, 1])
    assert [part.shape for part in parts] == [(60, 2), (60, 2)]
    merged = merge_multimodal_embeddings(
        torch.tensor([1, 7, 7, 2]), torch.zeros(4, 2),
        (torch.ones(2, 2),), [7],
    )
    assert torch.equal(merged, torch.tensor([[0., 0.], [1., 1.], [1., 1.], [0., 0.]]))
