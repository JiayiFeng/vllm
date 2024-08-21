import argparse
import os
from typing import List

import torch

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.inputs.data import (DistInfo, KVCacheBlobBase, PrefillKVCacheLoader,
                              PrefillKVCachePrompt)
from vllm.utils import FlexibleArgumentParser

current_file_path = os.path.abspath(__file__)


def get_dummy_kv_cache_input():
    return {
        "prompt":
        "The capital of China is ",
        "token_ids": [791, 6864, 315, 5734, 374, 220],
        "model_name":
        "NousResearch/Meta-Llama-3-8B-Instruct",
        "kv_cache":
        torch.randn((2, 32, 6, 8, 128), dtype=torch.bfloat16, device="cpu")
    }


class KVCacheBlob(KVCacheBlobBase):

    def __init__(self, kv_cache: torch.Tensor):
        self.kv_cache = kv_cache

    def blocks(self, dist_info: DistInfo,
               block_size: int) -> List[List[torch.Tensor]]:
        num_head = self.kv_cache.shape[3]
        num_tp_head = num_head // dist_info.tp_size
        start, end = dist_info.tp_rank * num_tp_head, (dist_info.tp_rank +
                                                       1) * num_tp_head
        rank_kv_cache = self.kv_cache[:, :, :, start:end, :]
        return [
            list(rank_kv_cache[:, layer_idx, :, :, :].split(block_size, dim=1))
            for layer_idx in range(rank_kv_cache.shape[1])
        ]


class PrefillKVCacheLoader(PrefillKVCacheLoader):

    def __init__(self, kv_cache: torch.Tensor):
        self.kv_cache = kv_cache

    def load(self) -> KVCacheBlob:
        return KVCacheBlob(self.kv_cache)

    def close(self):
        pass


def main(args: argparse.Namespace):
    inputs = get_dummy_kv_cache_input()
    query = PrefillKVCachePrompt(prompt_token_ids=inputs["token_ids"],
                                 kv_cache_loader=PrefillKVCacheLoader(
                                     inputs["kv_cache"][:, :, :-1, :, :]))
    args.model = inputs["model_name"]
    sampling_param = SamplingParams(temperature=0.0,
                                    logprobs=1,
                                    prompt_logprobs=1)
    engine_args = EngineArgs.from_cli_args(args)

    engine = LLMEngine.from_engine_args(engine_args)
    engine.add_request("0", inputs=query, params=sampling_param)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
