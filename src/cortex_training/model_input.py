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

"""Model inputs made of text and media chunks.

A prompt is an ordered list of chunks. Text arrives already tokenized; images,
video and audio arrive as encoded bytes (or a pointer to them) and the server
expands each one into the placeholder positions the model's processor needs.

The class names, fields and ``type`` values match the Tinker SDK's
``tinker.types`` so that code and renderers written for Tinker build the same
inputs here. Video has no Tinker counterpart; ``VideoChunk`` and
``VideoAssetPointerChunk`` are shaped like the image pair, plus an optional
``fps``.

Lengths count expanded positions. A media chunk's length is its
``expected_tokens``: the client states it (a renderer usually computes it from
the media size). A media chunk without ``expected_tokens`` has no known length,
so ``ModelInput.length`` raises for it rather than guessing. The service does not
take ``expected_tokens`` yet: /generate refuses a media chunk that sets it
(400 ``unsupported_prompt_shape``), so leave it unset in a prompt you send.

The types are frozen dataclasses (this package does not depend on pydantic).
``to_dict`` / ``ModelInput.from_dict`` convert to and from the chunk dicts;
with ``json=True`` media bytes become base64 text, and ``from_dict`` accepts
either form.

Experimental: generate accepts a ``ModelInput`` only with
``CORTEX_TRAINING_ENABLE_MEDIA_INPUT`` set to a truthy value. The service takes
``EncodedTextChunk``, ``ImageChunk`` and ``VideoChunk`` on /generate (media only
where it has media turned on) and refuses asset pointers and ``DmelChunk`` for
now; see "Chunked (``ModelInput``) prompts" in ``docs/reference/rest-api.md``.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Union

__all__ = [
    "DmelChunk",
    "EncodedTextChunk",
    "ImageAssetPointerChunk",
    "ImageChunk",
    "ModelInput",
    "ModelInputChunk",
    "VideoAssetPointerChunk",
    "VideoChunk",
]

_IMAGE_FORMATS = ("png", "jpeg")
_VIDEO_FORMATS = ("mp4",)
# Fields that hold encoded media bytes, base64 text in a JSON dict.
BYTE_FIELDS = ("data", "dmel")


def _as_bytes(value: Any, name: str) -> bytes:
    """Accept raw bytes, or base64 text as it appears in JSON."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"{name} must be bytes or base64 text") from exc
    raise TypeError(f"{name} must be bytes or base64 text, got {type(value).__name__}")


def _check_format(cls_name: str, value: Any, allowed: Tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{cls_name}.format must be one of {list(allowed)}, got {value!r}")


def _check_location(cls_name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{cls_name}.location must be a non-empty string")


def _check_fps(cls_name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{cls_name}.fps must be a positive number or None, got {value!r}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class _Chunk:
    type: ClassVar[str]

    def to_dict(self, *, json: bool = False) -> Dict[str, Any]:
        """This chunk as a dict with its ``type``; ``None`` fields are left out.

        ``json=True`` base64-encodes media bytes; otherwise they stay ``bytes``.
        """
        out: Dict[str, Any] = {"type": self.type}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name == "tokens":
                value = list(value)
            elif json and f.name in BYTE_FIELDS:
                value = base64.b64encode(value).decode("ascii")
            out[f.name] = value
        return out


@dataclass(frozen=True)
class EncodedTextChunk(_Chunk):
    """Token ids, already produced by the model's tokenizer."""

    tokens: Tuple[int, ...]
    type: ClassVar[str] = "encoded_text"

    def __post_init__(self) -> None:
        tokens = tuple(self.tokens)
        for i, token in enumerate(tokens):
            if not _is_int(token):
                raise ValueError(f"EncodedTextChunk.tokens[{i}] must be an integer token id, got {token!r}")
        object.__setattr__(self, "tokens", tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class _MediaChunk(_Chunk):
    def _check_expected_tokens(self) -> None:
        value = getattr(self, "expected_tokens")
        if value is not None and (not _is_int(value) or value < 1):
            raise ValueError(f"{type(self).__name__}.expected_tokens must be a positive integer or None, got {value!r}")

    @property
    def length(self) -> int:
        expected = getattr(self, "expected_tokens")
        if expected is None:
            raise ValueError(
                f"{type(self).__name__} has no expected_tokens, so its length is unknown; "
                "set expected_tokens (renderers compute it from the media size)"
            )
        return expected


@dataclass(frozen=True)
class ImageChunk(_MediaChunk):
    """An encoded image sent inline."""

    data: bytes
    format: str
    expected_tokens: Optional[int] = None
    type: ClassVar[str] = "image"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _as_bytes(self.data, "ImageChunk.data"))
        _check_format("ImageChunk", self.format, _IMAGE_FORMATS)
        self._check_expected_tokens()


@dataclass(frozen=True)
class ImageAssetPointerChunk(_MediaChunk):
    """An image the service reads from ``location`` instead of the request body."""

    format: str
    location: str
    expected_tokens: Optional[int] = None
    type: ClassVar[str] = "image_asset_pointer"

    def __post_init__(self) -> None:
        _check_format("ImageAssetPointerChunk", self.format, _IMAGE_FORMATS)
        _check_location("ImageAssetPointerChunk", self.location)
        self._check_expected_tokens()


@dataclass(frozen=True)
class VideoChunk(_MediaChunk):
    """An encoded video sent inline.

    ``fps`` is the rate frames are sampled at; ``None`` uses the model's default.
    The service does not take ``fps`` yet: /generate refuses a video chunk that
    sets it, and samples the video at the model processor's own rate. Leave it
    ``None``. For pre-extracted frames, send a run of ``ImageChunk``s instead.
    """

    data: bytes
    format: str
    fps: Optional[float] = None
    expected_tokens: Optional[int] = None
    type: ClassVar[str] = "video"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _as_bytes(self.data, "VideoChunk.data"))
        _check_format("VideoChunk", self.format, _VIDEO_FORMATS)
        _check_fps("VideoChunk", self.fps)
        self._check_expected_tokens()


@dataclass(frozen=True)
class VideoAssetPointerChunk(_MediaChunk):
    """A video the service reads from ``location`` instead of the request body."""

    format: str
    location: str
    fps: Optional[float] = None
    expected_tokens: Optional[int] = None
    type: ClassVar[str] = "video_asset_pointer"

    def __post_init__(self) -> None:
        _check_format("VideoAssetPointerChunk", self.format, _VIDEO_FORMATS)
        _check_location("VideoAssetPointerChunk", self.location)
        _check_fps("VideoAssetPointerChunk", self.fps)
        self._check_expected_tokens()


@dataclass(frozen=True)
class DmelChunk(_MediaChunk):
    """Audio already encoded on the client into quantized mel features.

    Only models with an audio encoder accept it.
    """

    dmel: bytes
    expected_tokens: Optional[int] = None
    type: ClassVar[str] = "dmel"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dmel", _as_bytes(self.dmel, "DmelChunk.dmel"))
        self._check_expected_tokens()


ModelInputChunk = Union[
    EncodedTextChunk,
    ImageChunk,
    ImageAssetPointerChunk,
    VideoChunk,
    VideoAssetPointerChunk,
    DmelChunk,
]

_CHUNK_TYPES: Dict[str, type] = {
    cls.type: cls
    for cls in (EncodedTextChunk, ImageChunk, ImageAssetPointerChunk, VideoChunk, VideoAssetPointerChunk, DmelChunk)
}


def _chunk_from_dict(value: Any, index: int) -> ModelInputChunk:
    if not isinstance(value, dict):
        raise TypeError(f"chunks[{index}] must be a dict, got {type(value).__name__}")
    fields = dict(value)
    kind = fields.pop("type", None)
    cls = _CHUNK_TYPES.get(kind)
    if cls is None:
        raise ValueError(f"chunks[{index}] has unknown type {kind!r}; expected one of {sorted(_CHUNK_TYPES)}")
    known = {f.name for f in dataclasses.fields(cls)}
    extra = sorted(set(fields) - known)
    if extra:
        raise ValueError(f"chunks[{index}] ({kind}) has unknown fields {extra}")
    try:
        return cls(**fields)
    except TypeError as exc:
        raise ValueError(f"chunks[{index}] ({kind}): {exc}") from exc


@dataclass(frozen=True)
class ModelInput:
    """An ordered sequence of text and media chunks."""

    chunks: Tuple[ModelInputChunk, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        chunks = tuple(self.chunks)
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, tuple(_CHUNK_TYPES.values())):
                raise TypeError(f"chunks[{i}] must be a ModelInput chunk, got {type(chunk).__name__}")
        object.__setattr__(self, "chunks", chunks)

    @classmethod
    def from_ints(cls, tokens: Iterable[int]) -> "ModelInput":
        return cls(chunks=(EncodedTextChunk(tokens=tuple(tokens)),))

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(chunks=())

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ModelInput":
        """Build from ``{"chunks": [...]}``; media bytes may be raw or base64 text."""
        if not isinstance(value, dict) or set(value) != {"chunks"} or not isinstance(value["chunks"], list):
            raise ValueError('a ModelInput dict is {"chunks": [...]} and nothing else')
        return cls(chunks=tuple(_chunk_from_dict(c, i) for i, c in enumerate(value["chunks"])))

    def to_dict(self, *, json: bool = False) -> Dict[str, Any]:
        """``{"chunks": [...]}``; ``json=True`` base64-encodes media bytes."""
        return {"chunks": [chunk.to_dict(json=json) for chunk in self.chunks]}

    def to_ints(self) -> List[int]:
        """Return the token ids; raises if any chunk is media."""
        tokens: List[int] = []
        for index, chunk in enumerate(self.chunks):
            if not isinstance(chunk, EncodedTextChunk):
                raise ValueError(f"chunks[{index}] is {chunk.type!r}, not text; to_ints needs a text-only input")
            tokens.extend(chunk.tokens)
        return tokens

    @property
    def length(self) -> int:
        """Total positions the input occupies, media expanded."""
        return sum(chunk.length for chunk in self.chunks)

    @property
    def has_media(self) -> bool:
        return any(not isinstance(chunk, EncodedTextChunk) for chunk in self.chunks)

    def append(self, chunk: ModelInputChunk) -> "ModelInput":
        return ModelInput(chunks=(*self.chunks, chunk))

    def append_int(self, token: int) -> "ModelInput":
        if self.chunks and isinstance(self.chunks[-1], EncodedTextChunk):
            last = self.chunks[-1]
            return ModelInput(chunks=(*self.chunks[:-1], EncodedTextChunk(tokens=(*last.tokens, token))))
        return self.append(EncodedTextChunk(tokens=(token,)))
