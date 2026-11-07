import argparse
import os

from src.models.qwen import base as qwen_base
from src.utils.misc import (
    read_jsonl,
    truncate_text,
    segment_text,
    format_prompt,
    load_config,
    save_config,
    get_new_experiment_path,
    load_prompt,
    parse_qa_response,
    save_seed_predictions,
)


def main(config):
    data_config = config["data_params"]
    eval_config = config["eval_params"]
    gen_config = config["generate_params"]
    seeds = gen_config["seeds"]
    per_seed_params = qwen_base.get_sampling_params(gen_config)
    if eval_config["segment_length"] <= 0 or eval_config["truncate_length"] < 0:
        raise ValueError(
            "segment_length must be positive and truncate_length nonnegative."
        )
    prompt_template = load_prompt(data_config["prompt_path"])
    data = read_jsonl(
        os.path.join(data_config["data_path"], eval_config["split"] + ".jsonl")
    )
    model_name = eval_config["checkpoint_path"] or eval_config["model_name"]
    model, tokenizer = qwen_base.load_vllm_model(
        model_name,
        gen_config["engine_seed"],
        enable_yarn=eval_config["enable_yarn"],
    )
    experiment_path = get_new_experiment_path(eval_config["save_path"])
    save_config(config, os.path.join(experiment_path, "config.yaml"))

    prompts, sampling_params, segment_counts = [], [], []
    for sample in data:
        context = sample["input"]
        if eval_config["truncate_length"] > 0:
            context = truncate_text(
                context, tokenizer, max_length=eval_config["truncate_length"]
            )
        segments = segment_text(
            context, tokenizer, max_length=eval_config["segment_length"]
        )
        segment_counts.append(len(segments))
        for segment in segments:
            user_prompt = format_prompt(
                prompt_template,
                book=sample["book"],
                character=sample["character"],
                context=segment,
            )
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=eval_config["thinking"],
            )
            prompts.extend([chat_prompt] * len(seeds))
            sampling_params.extend(per_seed_params)

    outputs = []
    for index in range(0, len(prompts), 10000):
        outputs.extend(
            qwen_base.generate_batch(
                model,
                prompts[index : index + 10000],
                sampling_params[index : index + 10000],
            )
        )

    fields = ("questions", "explanations", "answers", "types")
    predictions = []
    cursor = 0
    for sample, count in zip(data, segment_counts):
        prediction = {
            "book": sample["book"],
            "character": sample["character"],
            "seeds": seeds,
            **{field: [[] for _ in seeds] for field in fields},
        }
        for _ in range(count):
            for seed_index in range(len(seeds)):
                kept = [[], [], [], []]
                seen = set()
                for question, explanation, answer, qa_type in zip(
                    *parse_qa_response(outputs[cursor])
                ):
                    if (
                        question
                        and "none" not in question.lower()
                        and "none" not in answer.lower()
                        and (question, answer) not in seen
                    ):
                        for values, value in zip(
                            kept, (question, explanation, answer, qa_type)
                        ):
                            values.append(value)
                        seen.add((question, answer))
                cursor += 1
                for field, values in zip(fields, kept):
                    prediction[field][seed_index].append(values)
        predictions.append(prediction)
    save_seed_predictions(predictions, seeds, experiment_path, fields)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-d", required=True)
    args = parser.parse_args()
    main(load_config(args.config))
