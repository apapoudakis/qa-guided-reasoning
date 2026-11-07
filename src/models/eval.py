"""
Evaluate zero-shot or finetuned models in No-context, Lead, Hierarchical, and Incremental experiments.
Guided reasoning or built-in reasoning can be optionally enabled.
Configurations are specified in config/eval.yaml.
"""

import argparse
import os

from pathlib import Path

from src.models.qwen import base as qwen_base
from src.models.qwen.base import get_sampling_params, generate_batch
from src.utils.misc import (
    read_jsonl,
    truncate_text,
    segment_text,
    format_prompt,
    load_config,
    save_config,
    save_seed_predictions,
    get_new_experiment_path,
    load_template,
    parse_think_and_output,
)


def build_prompt(
    tokenizer,
    user_prompt: str,
    qa_content: str,
    guided_reasoning: bool,
    built_in_thinking: bool,
) -> str:
    """Build a complete prompt with optional thinking/QA content."""
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # keep it always True and control it bellow
    )

    if guided_reasoning:
        return chat_prompt + f"<think>\n{qa_content}\n</think>\n\n"
    elif not built_in_thinking:
        return chat_prompt + "<think>\n\n</think>\n\n"
    return chat_prompt


def format_qa_pairs(
    qa_item: dict, segment_idx, character, book, prefix_template, seed_idx
):
    """Format QA pairs into a string for guided reasoning."""
    if qa_item is None:
        return ""

    questions = qa_item["questions"][seed_idx]
    explanations = qa_item["explanations"][seed_idx]
    answers = qa_item["answers"][seed_idx]

    if segment_idx is not None:
        questions = questions[segment_idx]
        explanations = explanations[segment_idx]
        answers = answers[segment_idx]
    elif questions and isinstance(questions[0], list):
        questions = [x for sub in questions for x in sub]
        explanations = [x for sub in explanations for x in sub]
        answers = [x for sub in answers for x in sub]

    qa_lines = []
    existing_q = set()
    for q, e, a in zip(questions, explanations, answers):
        if q not in existing_q:
            qa_lines.append(f"Q: {q}\nExplanation: {e}\nA: {a}")

            existing_q.add(q)

    return prefix_template.format(character=character, book=book) + "\n".join(qa_lines)


def process_truncate(
    model, tokenizer, data, qa_data, prompt_template, eval_config, gen_config
):
    sampling_params = []
    num_seeds = len(gen_config["seeds"])

    qa_prefix = (
        "I need to describe {character} from the book {book} based on the given context. "
        "Let me start by going through the provided text and answering questions about {character} one by one.\n\n"
    )

    prompt_texts = []
    for i, sample in enumerate(data):
        context = ""
        if eval_config["method"] != "no_context":
            context = truncate_text(
                sample["input"], tokenizer, max_length=eval_config["truncate_length"]
            )
        user_prompt = format_prompt(
            prompt_template,
            book=sample["book"],
            character=sample["character"],
            context=context,
        )

        qa_str_per_seed = [""] * num_seeds
        if qa_data is not None:
            for s in range(num_seeds):
                qa_str_per_seed[s] = format_qa_pairs(
                    qa_data[i], None, sample["character"], sample["book"], qa_prefix, s
                )

        prompt_texts.extend(
            [
                build_prompt(
                    tokenizer,
                    user_prompt,
                    qa_str,
                    eval_config["guided_reasoning"],
                    eval_config["built_in_thinking"],
                )
                for qa_str in qa_str_per_seed
            ]
        )
        sampling_params.extend(get_sampling_params(gen_config))

    outputs = generate_batch(model, prompt_texts, sampling_params)

    results = []
    for i, s in enumerate(data):
        seed_outputs = outputs[i * num_seeds : (i + 1) * num_seeds]
        results.append(
            {"book": s["book"], "character": s["character"], "output": seed_outputs}
        )

    return results


def process_hierarchical(
    model, tokenizer, data, qa_data, prompt_template, eval_config, gen_config
):
    num_seeds = len(gen_config["seeds"])
    general_template, merge_template = prompt_template

    qa_prefix = (
        "I need to describe {character} from the book {book} based on the given context. "
        "Let me start by going through the provided text and answering questions about {character} one by one.\n "
    )

    results = []

    chunk_params = get_sampling_params(
        gen_config, max_new_tokens=gen_config["chunk_max_tokens"]
    )
    merge_params = get_sampling_params(gen_config)

    groups = []
    for i, sample in enumerate(data):
        if not groups or groups[-1][0] != sample["book"]:
            groups.append((sample["book"], []))
        groups[-1][1].append((i, sample))

    for book, items in groups:
        all_segment_data = []
        for i, sample in items:
            segments = segment_text(
                sample["input"],
                tokenizer,
                prefix="",
                max_length=eval_config["truncate_length"],
            )

            qa_strs = []
            for seg_idx, seg in enumerate(segments):
                qa_str = [""] * num_seeds
                if qa_data is not None:
                    for s in range(num_seeds):
                        qa_str[s] = format_qa_pairs(
                            qa_data[i],
                            seg_idx,
                            sample["character"],
                            sample["book"],
                            qa_prefix,
                            s,
                        )
                qa_strs.append(qa_str)

            all_segment_data.append((sample, segments, qa_strs))

        flat_prompts = []
        flat_sampling_params = []
        for sample, segments, qa_strs in all_segment_data:
            for seg, qa_str in zip(segments, qa_strs):
                for s in range(num_seeds):
                    user_prompt = format_prompt(
                        general_template,
                        book=sample["book"],
                        character=sample["character"],
                        context=seg,
                    )
                    prompt = build_prompt(
                        tokenizer,
                        user_prompt,
                        qa_str[s],
                        eval_config["guided_reasoning"],
                        eval_config["built_in_thinking"],
                    )
                    flat_prompts.append(prompt)
                    flat_sampling_params.append(chunk_params[s])

        flat_outputs = generate_batch(model, flat_prompts, flat_sampling_params)

        output_idx = 0
        for sample, segments, _ in all_segment_data:
            num_segments = len(segments)

            segment_outputs_per_seed = [[] for _ in range(num_seeds)]
            for seg_i in range(num_segments):
                for seed_idx in range(num_seeds):
                    segment_outputs_per_seed[seed_idx].append(flat_outputs[output_idx])
                    output_idx += 1

            merge_prompts = []
            merge_sampling_params = []
            for seed_idx in range(num_seeds):
                intermediate = ""
                for c, d in enumerate(segment_outputs_per_seed[seed_idx]):
                    if "<think>" in d:
                        think_output_dict = parse_think_and_output(d)
                        intermediate += f"Description from Chunk {c}: {think_output_dict['output']}\n\n"
                    else:
                        intermediate += f"Description from Chunk {c}: {d}\n\n"

                merge_prompt = format_prompt(
                    merge_template,
                    book=book,
                    character=sample["character"],
                    context=intermediate,
                )
                chat_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": merge_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=eval_config["hier_merge_thinking"],
                )
                merge_prompts.append(chat_prompt)
                merge_sampling_params.append(merge_params[seed_idx])

            merge_outputs = generate_batch(model, merge_prompts, merge_sampling_params)

            results.append(
                {
                    "book": book,
                    "character": sample["character"],
                    "segment_outputs": segment_outputs_per_seed,  # [seed_idx][segment_idx]
                    "output": merge_outputs,  # [seed_idx]
                }
            )

    return results


def process_incremental(
    model, tokenizer, data, qa_data, prompt_template, eval_config, gen_config
):
    num_seeds = len(gen_config["seeds"])
    chunk_params = get_sampling_params(
        gen_config, max_new_tokens=gen_config["chunk_max_tokens"]
    )
    first_template, update_template, final_template = prompt_template

    qa_prefix_first = (
        "I need to describe {character} from the book {book} based on the given context. "
        "Let me start by going through the provided text and answering questions about {character} one by one.\n "
    )
    qa_prefix_update = (
        "I need to update the current description of {character} from the book {book} based on the given context "
        "in case the character is mentioned. Let me start to understand the given context by going through "
        "the provided text and answering questions about {character} one by one.\n\n"
    )

    all_segments = [
        segment_text(
            sample["input"],
            tokenizer,
            prefix="",
            max_length=eval_config["truncate_length"],
        )
        for sample in data
    ]

    max_segments = max(len(segs) for segs in all_segments)

    intermediate_descriptions = [
        [[] for _ in range(num_seeds)] for _ in range(len(data))
    ]

    for seg_idx in range(max_segments):
        batch_prompts = []
        batch_sampling_params = []
        batch_metadata = []  # (sample_idx, seed_idx)

        for sample_idx, sample in enumerate(data):
            if seg_idx >= len(all_segments[sample_idx]):
                continue

            segment = all_segments[sample_idx][seg_idx]
            include_qa_flag = qa_data is not None

            for seed_idx in range(num_seeds):
                if seg_idx == 0:
                    # A lone chunk also needs the final description's length target.
                    template = (
                        final_template
                        if len(all_segments[sample_idx]) == 1
                        and final_template != update_template
                        else first_template
                    )
                    user_prompt = format_prompt(
                        template,
                        book=sample["book"],
                        character=sample["character"],
                        context=segment,
                        current_description="",
                    )
                    qa_str = (
                        format_qa_pairs(
                            qa_data[sample_idx],
                            0,
                            sample["character"],
                            sample["book"],
                            qa_prefix_first,
                            seed_idx,
                        )
                        if include_qa_flag
                        else ""
                    )
                else:
                    current_desc = (
                        intermediate_descriptions[sample_idx][seed_idx][-1]
                        if intermediate_descriptions[sample_idx][seed_idx]
                        else "None"
                    )
                    if "<think>" in current_desc:
                        current_desc = parse_think_and_output(current_desc)["output"]

                    template = (
                        final_template
                        if seg_idx == len(all_segments[sample_idx]) - 1
                        else update_template
                    )
                    user_prompt = format_prompt(
                        template,
                        book=sample["book"],
                        character=sample["character"],
                        context=segment,
                        current_description=current_desc,
                    )
                    qa_str = (
                        format_qa_pairs(
                            qa_data[sample_idx],
                            seg_idx,
                            sample["character"],
                            sample["book"],
                            qa_prefix_update,
                            seed_idx,
                        )
                        if include_qa_flag
                        else ""
                    )

                prompt = build_prompt(
                    tokenizer,
                    user_prompt,
                    qa_str,
                    eval_config["guided_reasoning"],
                    eval_config["built_in_thinking"],
                )
                batch_prompts.append(prompt)
                batch_sampling_params.append(chunk_params[seed_idx])
                batch_metadata.append((sample_idx, seed_idx))

        if batch_prompts:
            outputs = generate_batch(model, batch_prompts, batch_sampling_params)
            for i, (sample_idx, seed_idx) in enumerate(batch_metadata):
                intermediate_descriptions[sample_idx][seed_idx].append(outputs[i])

    results = []
    for i, sample in enumerate(data):
        results.append(
            {
                "book": sample["book"],
                "character": sample["character"],
                "intermediate_outputs": intermediate_descriptions[
                    i
                ],  # [seed_idx][step_idx]
                "output": [
                    intermediate_descriptions[i][seed_idx][-1]
                    if intermediate_descriptions[i][seed_idx]
                    else "None"
                    for seed_idx in range(num_seeds)
                ],
            }
        )

    return results


def validate_qa_data(data, qa_data, seeds, tokenizer, eval_config):
    if len(qa_data) != len(data):
        raise ValueError("Data and QA files must have the same number of rows.")
    chunked = eval_config["method"] in ("hierarchical", "incremental")
    for index, (sample, qa) in enumerate(zip(data, qa_data)):
        if (sample["book"], sample["character"]) != (qa["book"], qa["character"]):
            raise ValueError(f"QA record {index} does not match the input example.")
        if "seeds" in qa and qa["seeds"] != seeds:
            raise ValueError(
                f"QA record {index} has different inference seeds or seed order."
            )
        count = (
            len(
                segment_text(
                    sample["input"],
                    tokenizer,
                    max_length=eval_config["truncate_length"],
                )
            )
            if chunked
            else None
        )
        for field in ("questions", "explanations", "answers", "types"):
            values = qa[field]
            if len(values) != len(seeds):
                raise ValueError(
                    f"QA record {index} must have one {field} array per inference seed."
                )
            for chunks in values:
                if not isinstance(chunks, list) or any(
                    not isinstance(chunk, list) for chunk in chunks
                ):
                    raise ValueError(
                        "Use the combined QA preds.jsonl with seed/chunk/QA dimensions."
                    )
                if chunked and len(chunks) != count:
                    raise ValueError(
                        f"QA record {index} must have one {field} list per input chunk."
                    )
        for seed_index in range(len(seeds)):
            fields = [
                qa[field][seed_index]
                for field in ("questions", "explanations", "answers", "types")
            ]
            if len({len(field) for field in fields}) != 1:
                raise ValueError(f"QA record {index} has unaligned chunk arrays.")
            for chunks in zip(*fields):
                if len({len(chunk) for chunk in chunks}) != 1:
                    raise ValueError(f"QA record {index} has unaligned QA fields.")


def main(config):
    data_config = config["data_params"]
    eval_config = config["eval_params"]
    gen_config = config["generate_params"]
    method = eval_config["method"]
    if method not in ("truncate", "no_context", "hierarchical", "incremental", None):
        raise ValueError(f"Unknown generation method: {method}")
    get_sampling_params(gen_config)
    if eval_config["guided_reasoning"] and not data_config["qa_data_path"]:
        raise ValueError("guided_reasoning requires qa_data_path.")
    if method != "no_context" and eval_config["truncate_length"] <= 0:
        raise ValueError("truncate_length must be positive.")
    prompt_template = load_template(data_config["prompt_path"], method=method)
    data = read_jsonl(
        os.path.join(data_config["data_path"], eval_config["split"] + ".jsonl")
    )
    qa_data = None
    if eval_config["guided_reasoning"]:
        qa_path = Path(data_config["qa_data_path"])
        qa_data = read_jsonl(qa_path / "preds.jsonl")
        saved_config = qa_path / "config.yaml"
        if saved_config.exists():
            qa_seeds = load_config(saved_config)["generate_params"]["seeds"]
            if qa_seeds != gen_config["seeds"]:
                raise ValueError(
                    "QA and description inference seeds must match in the same order."
                )

    checkpoint = eval_config["checkpoint_path"] or eval_config["model_name"]
    tokenizer = qwen_base.AutoTokenizer.from_pretrained(checkpoint)
    if qa_data is not None:
        validate_qa_data(data, qa_data, gen_config["seeds"], tokenizer, eval_config)
    model, tokenizer = qwen_base.load_vllm_model(
        checkpoint,
        gen_config["engine_seed"],
        enable_yarn=eval_config["enable_yarn"],
        tokenizer=tokenizer,
    )
    experiment_path = get_new_experiment_path(eval_config["save_path"])
    save_config(config, os.path.join(experiment_path, "config.yaml"))
    if not data:
        results = []
    elif method in ("truncate", "no_context", None):
        results = process_truncate(
            model, tokenizer, data, qa_data, prompt_template, eval_config, gen_config
        )
    elif method == "hierarchical":
        results = process_hierarchical(
            model, tokenizer, data, qa_data, prompt_template, eval_config, gen_config
        )
    else:
        results = process_incremental(
            model, tokenizer, data, qa_data, prompt_template, eval_config, gen_config
        )
    for result in results:
        result["seeds"] = gen_config["seeds"]
    save_seed_predictions(
        results,
        gen_config["seeds"],
        experiment_path,
        ("output", "segment_outputs", "intermediate_outputs"),
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-d", required=True)
    args = parser.parse_args()
    main(load_config(args.config))
