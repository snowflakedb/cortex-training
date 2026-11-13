# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The supervision mask a DFlash2 forward-backward request carries."""

from __future__ import annotations

from recipes.dflash2.task import TokenizedConversation
from recipes.dflash2.task import collate


def test_the_request_mask_supervises_exactly_the_response_tokens():
    # The dflash2 loss reads entry i as the supervision of token i + 1, so the response [A, B] after two prompt
    # tokens is [0, 1, 1, 0]: A and B are supervised, and nothing after the response is.
    response = TokenizedConversation(input_ids=[10, 11, 12, 13], loss_mask=[0, 0, 1, 1])
    shorter = TokenizedConversation(input_ids=[20, 21, 22], loss_mask=[0, 1, 1])

    kwargs, context = collate([response, shorter], pad_token_id=0, max_seq_len=8)

    assert context["loss_mask"].tolist() == [[0, 1, 1, 0], [1, 1, 0, 0]]
    assert kwargs["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]


def test_the_request_mask_keeps_every_supervised_token_but_the_first_of_a_row():
    # The first token of a row has no logits predicting it, so it cannot be supervised.
    seq = TokenizedConversation(input_ids=[1, 2, 3, 4, 5], loss_mask=[1, 1, 0, 1, 1])

    _, context = collate([seq], pad_token_id=0, max_seq_len=8)

    assert context["loss_mask"].tolist() == [[1, 0, 1, 1, 0]]
    assert context["loss_mask"].sum().item() == seq.num_supervised - 1
