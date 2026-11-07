import os

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_vllm_model(model_name, seed, enable_yarn=False, tokenizer=None, **model_params):
    """Load a vLLM model and tokenizer using the visible CUDA GPUs."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("vLLM inference requires a CUDA GPU.")
    model_params.setdefault("tensor_parallel_size", num_gpus)

    if "1M" in model_name:
        model_options = {
            "max_model_len": 1010000,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 131072,
            "enforce_eager": True,
        }
    else:
        model_options = {"enable_prefix_caching": True, "enforce_eager": True}
        if enable_yarn:
            model_options.update(
                rope_scaling={
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                },
                max_model_len=131072,
            )
            model_options.pop("enforce_eager")

    model_options.update(model_params)
    model = LLM(model=model_name, seed=seed, **model_options)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_sampling_params(generation_params, max_new_tokens=None, min_new_tokens=None):
    """Create one sampling configuration per inference seed."""
    seeds = generation_params["seeds"]
    if (
        not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must contain distinct integers.")
    return [
        SamplingParams(
            temperature=generation_params["temperature"],
            frequency_penalty=generation_params["frequency_penalty"],
            max_tokens=generation_params["max_new_tokens"]
            if max_new_tokens is None
            else max_new_tokens,
            min_tokens=generation_params["min_new_tokens"]
            if min_new_tokens is None
            else min_new_tokens,
            seed=seed,
        )
        for seed in seeds
    ]


def generate_batch(model, prompts, sampling_params):
    """Return one completion per prompt, preserving input order."""
    if not prompts:
        return []
    outputs = model.generate(prompts, sampling_params)
    if len(outputs) != len(prompts) or any(
        len(output.outputs) != 1 for output in outputs
    ):
        raise ValueError("The model must return one completion per prompt.")
    return [output.outputs[0].text for output in outputs]


def vllm_prompt_model(all_prompts, tokenizer, model, generation_params, thinking=False):
    """Generate text for a batch of user prompts with the chat template."""
    sampling_params = SamplingParams(
        temperature=generation_params["temperature"],
        frequency_penalty=generation_params["frequency_penalty"],
        max_tokens=generation_params["max_new_tokens"],
        min_tokens=generation_params["min_new_tokens"],
    )

    prompt_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        for prompt in all_prompts
    ]

    outputs = model.generate(prompt_texts, sampling_params)

    return [completion.text for output in outputs for completion in output.outputs]
