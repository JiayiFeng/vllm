import argparse
import os
from typing import List

import torch

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.inputs.data import PrefillKVCachePrompt
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


def main(args: argparse.Namespace):
    inputs = get_dummy_kv_cache_input()
    query = PrefillKVCachePrompt(prompt=inputs["prompt"],
                                 prompt_token_ids=inputs["token_ids"],
                                 kv_cache=inputs["kv_cache"])
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
