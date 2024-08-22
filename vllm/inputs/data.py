import dataclasses
from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, List, Literal, Optional, Sequence,
                    TypedDict, Union, cast, overload, Callable, NamedTuple)

from typing_extensions import NotRequired

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict
    import torch


class ParsedText(TypedDict):
    content: str
    is_tokens: Literal[False]


class ParsedTokens(TypedDict):
    content: List[int]
    is_tokens: Literal[True]


# https://github.com/vllm-project/vllm/pull/4028
@overload
def parse_and_batch_prompt(
        prompt: Union[str, List[str]]) -> Sequence[ParsedText]:
    ...


@overload
def parse_and_batch_prompt(
        prompt: Union[List[int], List[List[int]]]) -> Sequence[ParsedTokens]:
    ...


def parse_and_batch_prompt(
        prompt: Union[str, List[str], List[int], List[List[int]]],
) -> Union[Sequence[ParsedText], Sequence[ParsedTokens]]:
    if isinstance(prompt, str):
        # case 1: a string
        return [ParsedText(content=prompt, is_tokens=False)]

    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")

        if isinstance(prompt[0], str):
            # case 2: array of strings
            return [
                ParsedText(content=elem, is_tokens=False)
                for elem in cast(List[str], prompt)
            ]
        if isinstance(prompt[0], int):
            # case 3: array of tokens
            elem = cast(List[int], prompt)
            return [ParsedTokens(content=elem, is_tokens=True)]
        if isinstance(prompt[0], list):
            if len(prompt[0]) == 0:
                raise ValueError("please provide at least one prompt")

            if isinstance(prompt[0][0], int):
                # case 4: array of token arrays
                return [
                    ParsedTokens(content=elem, is_tokens=True)
                    for elem in cast(List[List[int]], prompt)
                ]

    raise ValueError("prompt must be a string, array of strings, "
                     "array of tokens, or array of token arrays")


class TextPrompt(TypedDict):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


class KVCacheLoadTileArgs(NamedTuple):
    tp_size: int
    tp_rank: int
    num_layers: int
    block_size: int
    num_heads: int
    head_dim: int


class KVCacheBlob(NamedTuple):
    blocks: List[List["torch.Tensor"]]
    close: Callable[[], None]


# input param: dist_info, block_shape, num_layers
PrefillKVCacheLoader = Callable[[KVCacheLoadTileArgs], KVCacheBlob]


class PrefillKVCachePrompt(TypedDict):
    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    kv_cache_loader: PrefillKVCacheLoader


PromptInputs = Union[str, TextPrompt, TokensPrompt, PrefillKVCachePrompt]
"""
The inputs to the LLM, which can take one of the following forms:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
"""


class LLMInputs(TypedDict):
    """
    The inputs in :class:`~vllm.LLMEngine` before they are
    passed to the model executor.
    """
    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    prompt: NotRequired[Optional[str]]
    """
    The original prompt text corresponding to the token IDs, if available.
    """

    multi_modal_data: NotRequired[Optional["MultiModalDataDict"]]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    kv_cache_loader: NotRequired[Optional[PrefillKVCacheLoader]]
