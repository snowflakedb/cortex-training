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

"""Unit tests for ``cortex_training.model_input`` (Tinker-shaped text and media chunks)."""

from __future__ import annotations

import base64
import dataclasses

import pytest

from cortex_training.model_input import (
    DmelChunk,
    EncodedTextChunk,
    ImageAssetPointerChunk,
    ImageChunk,
    ModelInput,
    VideoAssetPointerChunk,
    VideoChunk,
)

PNG = b"\x89PNG\r\n\x1a\nfake"
MP4 = b"\x00\x00\x00\x18ftypmp42fake"


def _mixed() -> ModelInput:
    return ModelInput(
        chunks=[
            EncodedTextChunk(tokens=[1, 2, 3]),
            ImageChunk(data=PNG, format="png", expected_tokens=64),
            EncodedTextChunk(tokens=[4]),
            VideoChunk(data=MP4, format="mp4", fps=2.0, expected_tokens=256),
            EncodedTextChunk(tokens=[5, 6]),
        ]
    )


def test_type_values_match_tinker():
    assert EncodedTextChunk(tokens=[1]).type == "encoded_text"
    assert ImageChunk(data=PNG, format="png").type == "image"
    assert ImageAssetPointerChunk(format="jpeg", location="a.jpg").type == "image_asset_pointer"
    assert DmelChunk(dmel=b"x").type == "dmel"
    assert VideoChunk(data=MP4, format="mp4").type == "video"
    assert VideoAssetPointerChunk(format="mp4", location="v.mp4").type == "video_asset_pointer"


def test_from_ints_to_ints_round_trip():
    mi = ModelInput.from_ints([7, 8, 9])
    assert mi.to_ints() == [7, 8, 9]
    assert mi.length == 3
    assert not mi.has_media


def test_to_ints_refuses_media():
    with pytest.raises(ValueError, match=r"chunks\[1\] is 'image'"):
        _mixed().to_ints()


def test_length_counts_expanded_media_positions():
    assert _mixed().length == 3 + 64 + 1 + 256 + 2


def test_length_needs_expected_tokens():
    mi = ModelInput(chunks=[EncodedTextChunk(tokens=[1]), ImageChunk(data=PNG, format="png")])
    assert mi.has_media
    with pytest.raises(ValueError, match="expected_tokens"):
        _ = mi.length


def test_append_returns_new_input_and_append_int_extends_text():
    mi = ModelInput.empty()
    mi2 = mi.append(EncodedTextChunk(tokens=[1]))
    assert mi.chunks == () and mi2.length == 1
    mi3 = mi2.append_int(2)
    assert len(mi3.chunks) == 1 and mi3.chunks[0].tokens == (1, 2)
    mi4 = mi3.append(ImageChunk(data=PNG, format="png", expected_tokens=4)).append_int(9)
    assert [c.type for c in mi4.chunks] == ["encoded_text", "image", "encoded_text"]


def test_inputs_are_frozen_and_compare_by_value():
    mi = _mixed()
    with pytest.raises(dataclasses.FrozenInstanceError):
        mi.chunks = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        mi.chunks[1].data = b""  # type: ignore[misc]
    assert mi == _mixed()
    assert hash(mi) == hash(_mixed())


def test_json_round_trip_base64_encodes_bytes():
    mi = _mixed()
    payload = mi.to_dict(json=True)
    image = payload["chunks"][1]
    assert image == {"type": "image", "data": base64.b64encode(PNG).decode("ascii"), "format": "png",
                     "expected_tokens": 64}
    assert ModelInput.from_dict(payload) == mi


def test_python_dict_keeps_raw_bytes_and_leaves_out_none():
    chunks = _mixed().to_dict()["chunks"]
    assert chunks[1]["data"] == PNG
    assert chunks[0] == {"type": "encoded_text", "tokens": [1, 2, 3]}
    assert "fps" not in ImageAssetPointerChunk(format="png", location="x").to_dict()
    assert "expected_tokens" not in VideoChunk(data=MP4, format="mp4").to_dict()
    assert ModelInput.from_dict(_mixed().to_dict()) == _mixed()


@pytest.mark.parametrize(
    "chunk",
    [
        {"type": "image", "data": "not base64!", "format": "png"},
        {"type": "image", "data": "aGk=", "format": "gif"},
        {"type": "video", "data": "aGk=", "format": "mp4", "fps": 0},
        {"type": "video", "data": "aGk=", "format": "mp4", "fps": float("nan")},
        {"type": "image", "data": "aGk=", "format": "png", "expected_tokens": 0},
        {"type": "image", "data": "aGk=", "format": "png", "expected_tokens": True},
        {"type": "image_asset_pointer", "format": "png", "location": ""},
        {"type": "image", "data": "aGk=", "format": "png", "width": 10},
        {"type": "image", "format": "png"},
        {"type": "encoded_text", "tokens": [1, "2"]},
        {"type": "audio", "data": "aGk="},
    ],
)
def test_invalid_chunks_are_rejected(chunk):
    with pytest.raises(ValueError):
        ModelInput.from_dict({"chunks": [chunk]})


def test_from_dict_needs_the_chunks_shape():
    with pytest.raises(ValueError, match="chunks"):
        ModelInput.from_dict({"chunks": [], "extra": 1})
    with pytest.raises(TypeError):
        ModelInput(chunks=[{"type": "encoded_text", "tokens": [1]}])  # type: ignore[list-item]
