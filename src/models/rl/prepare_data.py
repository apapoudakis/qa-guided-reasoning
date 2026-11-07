import argparse
import random

from transformers import AutoTokenizer

from src.utils.misc import (
    load_config,
    load_prompt,
    parse_qa_response,
    read_jsonl,
    segment_text,
    write_jsonl,
)


def format_ground_truth(description, qa):
    """Format the global reference QA pairs shared by a character's chunks."""
    if "questions" in qa:
        fields = tuple(
            qa[field] for field in ("questions", "explanations", "answers", "types")
        )
    else:
        fields = parse_qa_response(qa["qa_pairs"])
    if not fields[0] or any(len(field) != len(fields[0]) for field in fields):
        raise ValueError(
            "Each reference record must contain aligned, nonempty QA pairs."
        )
    if any(not isinstance(value, str) for field in fields for value in field):
        raise ValueError(
            "Reference QA must be flat lists, without seed or chunk dimensions."
        )
    qa_text = "".join(
        f"Q{index}: {question}\nE{index}: {explanation}\nA{index}: {answer}\nT{index}: {qa_type}\n\n"
        for index, (question, explanation, answer, qa_type) in enumerate(
            zip(*fields), start=1
        )
    )
    return f"Description: {description}\n\nQA Pairs:\n{qa_text}"


def prepare_data(
    data_path,
    save_path,
    prompt_path,
    model_name,
    qa_data_path,
    segment_length=16384,
    method="segment",
    seed=0,
):
    """Sample up to two chunks per character with global QA supervision."""
    if method != "segment":
        raise ValueError("Only the segment method is supported.")
    if type(segment_length) is not int or segment_length <= 0:
        raise ValueError("Segment length must be a positive integer.")
    prompt_template = load_prompt(prompt_path)
    data = read_jsonl(data_path)
    qa_data = read_jsonl(qa_data_path)
    if len(qa_data) != len(data):
        raise ValueError(
            "Data and reference QA files must have the same number of rows."
        )
    labels = []
    for sample, qa in zip(data, qa_data):
        if (sample["book"], sample["character"]) != (qa["book"], qa["character"]):
            raise ValueError("Reference QA records do not match the input order.")
        labels.append(format_ground_truth(sample["description"], qa))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rng = random.Random(seed)
    processed_data = []
    for sample, ground_truth in zip(data, labels):
        segments = segment_text(sample["input"], tokenizer, max_length=segment_length)
        if len(segments) >= 2:
            segments = rng.sample(segments, 2)
        for segment in segments:
            user_prompt = prompt_template.format(
                book=sample["book"],
                character=sample["character"],
                context=segment,
            )
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            processed_data.append(
                {
                    "input": chat_prompt,
                    "ground_truth": ground_truth,
                    "book": sample["book"],
                    "character": sample["character"],
                }
            )
    write_jsonl(processed_data, save_path)
    print(f"Prepared {len(processed_data)} chunks with global reference QA pairs.")
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare_data(**load_config(args.config))
