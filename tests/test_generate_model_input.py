# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ModelInput prompts (text and media chunks) in the generate request frame."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from cortex_training import wire
from cortex_training.client import core as nc
from cortex_training.model_input import (
    DmelChunk,
    EncodedTextChunk,
    ImageAssetPointerChunk,
    ImageChunk,
    ModelInput,
    VideoAssetPointerChunk,
    VideoChunk,
)

PNG = b"\x89PNG\r\n\x1a\nfake-image"
MP4 = b"\x00\x00\x00\x18ftypmp42fake-video"
DMEL = bytes(range(16))


@pytest.fixture(autouse=True)
def _media_input_enabled(monkeypatch):
    monkeypatch.setenv(nc.MEDIA_INPUT_ENV, "1")


def _response(body):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


def _client(post_json=None, max_seq_len=None) -> nc.CortexTrainingClient:
    client = nc.CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    client._session.post.return_value = _response(post_json or {"request_id": "r1", "count": 1})
    job = {}
    if max_seq_len is not None:
        job = {"sub_jobs": [{"job_type": "sampling", "inference_config": {"max_seq_len": float(max_seq_len)}}]}
    client._session.get.return_value = _response(job)
    return client


def _sent_prompts(client):
    return wire.loads(client._session.post.call_args.kwargs["data"])["prompts"]


def _every_chunk() -> ModelInput:
    return ModelInput(
        chunks=[
            EncodedTextChunk(tokens=[1, 2, 3]),
            ImageChunk(data=PNG, format="png", expected_tokens=64),
            ImageAssetPointerChunk(format="jpeg", location="@DB.SCH.STAGE/cat.jpg", expected_tokens=16),
            VideoChunk(data=MP4, format="mp4", fps=2.0, expected_tokens=256),
            VideoAssetPointerChunk(format="mp4", location="@DB.SCH.STAGE/clip.mp4"),
            DmelChunk(dmel=DMEL, expected_tokens=8),
            EncodedTextChunk(tokens=[4, 5]),
        ]
    )


def _as_bytes(tensor) -> bytes:
    assert tensor.dtype == torch.uint8 and tensor.dim() == 1
    return tensor.numpy().tobytes()


class TestExperimentalGate:
    def test_model_input_is_refused_by_default(self, monkeypatch):
        monkeypatch.delenv(nc.MEDIA_INPUT_ENV)
        c = _client()
        with pytest.raises(ValueError, match=nc.MEDIA_INPUT_ENV):
            c.generate("j1", prompts=[ModelInput.from_ints([1])])
        with pytest.raises(ValueError, match=nc.MEDIA_INPUT_ENV):
            c.generate_stream("j1", prompts=["hi", ModelInput.from_ints([1])])
        c._session.post.assert_not_called()

    def test_a_falsy_value_keeps_it_off(self, monkeypatch):
        monkeypatch.setenv(nc.MEDIA_INPUT_ENV, "0")
        c = _client()
        with pytest.raises(ValueError, match=nc.MEDIA_INPUT_ENV):
            c.generate("j1", prompts=ModelInput.from_ints([1]))
        c._session.post.assert_not_called()

    def test_plain_prompts_are_not_gated(self, monkeypatch):
        monkeypatch.delenv(nc.MEDIA_INPUT_ENV)
        c = _client()
        assert c.generate("j1", prompts=["hi", [1, 2]]) == "r1"


class TestModelInputFrame:
    def test_every_chunk_type_round_trips_through_the_frame(self):
        c = _client()
        c.generate("j1", prompts=[_every_chunk()])
        [prompt] = _sent_prompts(c)
        chunks = prompt["chunks"]
        assert [ch["type"] for ch in chunks] == [
            "encoded_text",
            "image",
            "image_asset_pointer",
            "video",
            "video_asset_pointer",
            "dmel",
            "encoded_text",
        ]
        text, image, image_ptr, video, video_ptr, dmel, tail = chunks
        assert text["tokens"].dtype == torch.int32 and text["tokens"].tolist() == [1, 2, 3]
        assert tail["tokens"].tolist() == [4, 5]
        assert _as_bytes(image["data"]) == PNG
        assert image["format"] == "png" and image["expected_tokens"] == 64
        assert image_ptr == {
            "type": "image_asset_pointer",
            "format": "jpeg",
            "location": "@DB.SCH.STAGE/cat.jpg",
            "expected_tokens": 16,
        }
        assert _as_bytes(video["data"]) == MP4 and video["fps"] == 2.0
        # None fields are left out rather than sent as null.
        assert video_ptr == {"type": "video_asset_pointer", "format": "mp4", "location": "@DB.SCH.STAGE/clip.mp4"}
        assert _as_bytes(dmel["dmel"]) == DMEL and dmel["expected_tokens"] == 8

    def test_decoded_chunks_rebuild_the_same_model_input(self):
        original = _every_chunk()
        c = _client()
        c.generate("j1", prompts=[original])
        [prompt] = _sent_prompts(c)

        def plain(value):
            if torch.is_tensor(value):
                return value.numpy().tobytes() if value.dtype == torch.uint8 else value.tolist()
            return value

        rebuilt = ModelInput.from_dict({"chunks": [{k: plain(v) for k, v in ch.items()} for ch in prompt["chunks"]]})
        assert rebuilt == original

    def test_single_model_input_is_one_prompt(self):
        c = _client()
        c.generate("j1", prompts=ModelInput.from_ints([7, 8]))
        prompt = _sent_prompts(c)
        assert isinstance(prompt, dict)
        assert prompt["chunks"][0]["tokens"].tolist() == [7, 8]

    def test_mixed_batch_keeps_strings_and_token_lists(self):
        c = _client()
        c.generate("j1", prompts=["hello", [1, 2], _every_chunk()])
        text, tokens, media = _sent_prompts(c)
        assert text == "hello"
        assert tokens.dtype == torch.int32 and tokens.tolist() == [1, 2]
        assert media["chunks"][1]["type"] == "image"

    def test_stream_frame_matches_generate_frame(self):
        unary = _client()
        unary.generate("j1", prompts=[_every_chunk()])
        stream = _client()
        stream.generate_stream("j1", prompts=[_every_chunk()])
        # Compared decoded: safetensors does not keep header metadata key order.
        [from_unary] = _sent_prompts(unary)
        [from_stream] = _sent_prompts(stream)
        assert len(from_unary["chunks"]) == len(from_stream["chunks"])
        for a, b in zip(from_unary["chunks"], from_stream["chunks"]):
            assert a.keys() == b.keys()
            for key in a:
                if torch.is_tensor(a[key]):
                    assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key])
                else:
                    assert a[key] == b[key]

    def test_disabled_tensor_prompts_keep_tokens_as_lists_but_media_as_bytes(self, monkeypatch):
        monkeypatch.setenv(nc.DISABLE_TENSOR_PROMPTS_ENV, "1")
        c = _client()
        c.generate("j1", prompts=[_every_chunk()])
        [prompt] = _sent_prompts(c)
        assert prompt["chunks"][0]["tokens"] == [1, 2, 3]
        assert _as_bytes(prompt["chunks"][1]["data"]) == PNG

    def test_empty_media_bytes_are_sent_for_the_server_to_reject(self):
        c = _client()
        c.generate("j1", prompts=[ModelInput(chunks=[ImageChunk(data=b"", format="png", expected_tokens=1)])])
        [prompt] = _sent_prompts(c)
        data = prompt["chunks"][0]["data"]
        assert data.dtype == torch.uint8 and data.numel() == 0

    def test_large_media_is_request_chunked_and_reassembles(self):
        c = _client()
        c._MAX_GENERATE_BYTES = 64 * 1024
        c._session.post.side_effect = [_response({"chunk_cached": True}), _response({"request_id": "big"})]
        payload = bytes(range(256)) * 400  # ~100 KiB, over the cap above
        prompt = ModelInput(chunks=[EncodedTextChunk(tokens=[1]), VideoChunk(data=payload, format="mp4")])

        assert c.generate("j1", prompts=[prompt]) == "big"
        frames = [call.kwargs["data"] for call in c._session.post.call_args_list]
        assert len(frames) == 2
        decoded = wire.loads(wire.decode_byte_chunks(frames, kind="request"))
        assert _as_bytes(decoded["prompts"][0]["chunks"][1]["data"]) == payload

    def test_stream_rejects_oversized_media(self):
        c = _client()
        c._MAX_GENERATE_BYTES = 1024
        prompt = ModelInput(chunks=[VideoChunk(data=b"x" * 4096, format="mp4")])
        with pytest.raises(ValueError, match="exceeds the maximum allowed size"):
            c.generate_stream("j1", prompts=[prompt])
        c._session.post.assert_not_called()


class TestModelInputLengthCheck:
    def test_text_plus_expected_tokens_that_fill_the_window_are_rejected(self):
        c = _client(max_seq_len=70)
        prompt = ModelInput(
            chunks=[EncodedTextChunk(tokens=[1] * 6), ImageChunk(data=PNG, format="png", expected_tokens=64)]
        )
        with pytest.raises(ValueError, match=r"occupies 70 positions .* max_seq_len of 70"):
            c.generate("j1", prompts=[prompt])
        c._session.post.assert_not_called()

    def test_prompt_under_the_window_is_sent(self):
        c = _client(max_seq_len=71)
        prompt = ModelInput(
            chunks=[EncodedTextChunk(tokens=[1] * 6), ImageChunk(data=PNG, format="png", expected_tokens=64)]
        )
        assert c.generate("j1", prompts=[prompt]) == "r1"

    def test_unknown_media_length_is_a_lower_bound(self):
        # The pointer has no expected_tokens: 3 known positions fit a window of 4,
        # so the client cannot know it overflows and leaves it to the server.
        c = _client(max_seq_len=4)
        prompt = ModelInput(
            chunks=[EncodedTextChunk(tokens=[1, 2, 3]), ImageAssetPointerChunk(format="png", location="@S/a.png")]
        )
        assert c.generate("j1", prompts=[prompt]) == "r1"

    def test_lower_bound_that_already_fills_the_window_is_rejected(self):
        c = _client(max_seq_len=4)
        prompt = ModelInput(
            chunks=[EncodedTextChunk(tokens=[1, 2, 3, 4]), ImageAssetPointerChunk(format="png", location="@S/a.png")]
        )
        with pytest.raises(ValueError, match="occupies at least 4 positions"):
            c.generate_stream("j1", prompts=prompt)

    def test_index_names_the_offending_prompt_in_a_mixed_batch(self):
        c = _client(max_seq_len=8)
        long_media = ModelInput(chunks=[VideoChunk(data=MP4, format="mp4", expected_tokens=8)])
        with pytest.raises(ValueError, match="prompt at index 2"):
            c.generate("j1", prompts=["text", [1, 2], long_media])
