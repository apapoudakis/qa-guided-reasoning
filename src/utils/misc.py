import json
import os
import re
import string
import unicodedata
from itertools import islice
from pathlib import Path

import yaml
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from unidecode import unidecode

_SENT_SPAN_TOKENIZER = PunktSentenceTokenizer().span_tokenize


def read_jsonl(file_path):
    data = []
    with open(file_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {file_path}, line {line_number}") from error
    return data


def write_jsonl(data, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        for row in data:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(config_path):
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_config(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def save_seed_predictions(predictions, seeds, experiment_path, fields):
    """Save combined predictions and an ordinary predictions file per seed."""
    write_jsonl(predictions, Path(experiment_path) / "preds.jsonl")
    for index, seed in enumerate(seeds):
        rows = []
        for prediction in predictions:
            row = {"book": prediction["book"], "character": prediction["character"]}
            row.update({field: prediction[field][index] for field in fields if field in prediction})
            rows.append(row)
        write_jsonl(rows, Path(experiment_path) / f"seed_{seed}" / "preds.jsonl")


def parse_qa_response(response: str):
    """
    Parse responses with flexible formats where only Q is required:
    - Q             : Q1: <question>
    - Q A           : Q1: <question> A1: <answer>
    - Q E           : Q1: <question> E1: <explanation>
    - Q T           : Q1: <question> T1: <type>
    - Q E A         : Q1: <question> E1: <explanation> A1: <answer>
    - Q A T         : Q1: <question> A1: <answer> T1: <type>
    - Q E T         : Q1: <question> E1: <explanation> T1: <type>
    - Q E A T       : Q1: <question> E1: <explanation> A1: <answer> T1: <type>
    """

    pattern = re.compile(
        r"Q\d+:\s*(.*?)\s*"
        r"(?:E\d+:\s*(.*?)\s*)?"      # Optional explanation
        r"(?:A\d+:\s*(.*?)\s*)?"      # Optional answer
        r"(?:T\d+:\s*(.*?))?"         # Optional type
        r"(?=\s*Q\d+:|$)",
        re.DOTALL,
    )

    questions, explanations, answers, types = [], [], [], []

    for match in pattern.finditer(response):
        questions.append(match.group(1).strip())
        explanations.append(match.group(2).strip() if match.group(2) else "")
        answers.append(match.group(3).strip() if match.group(3) else "")
        types.append(match.group(4).strip() if match.group(4) else "")

    return questions, explanations, answers, types

def normalize(s):
    s = unicodedata.normalize("NFD", s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return unidecode(white_space_fix(remove_articles(remove_punc(lower(s)))))

def count_tokens(text):
    return len(word_tokenize(text))

def truncate_text(text: str, tokenizer, max_length: int = 4096) -> str:

    current_len = 0
    last_end = 0

    for start, end in _SENT_SPAN_TOKENIZER(text):
        sent = text[start:end]
        sent_len = len(tokenizer.encode(sent))

        if current_len + sent_len > max_length:
            break

        current_len += sent_len
        last_end = end

    if last_end == 0:
        return tokenizer.decode(tokenizer.encode(text)[:max_length])

    return text[:last_end]

def segment_text(text, tokenizer, prefix="", max_length=4096):
    paragraphs = text.split("\n")

    pref_ids = tokenizer.encode(prefix, add_special_tokens=False)
    pref_len = len(pref_ids)

    enc = tokenizer(
        paragraphs,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_length=True,
    )
    if "length" in enc:
        lens = enc["length"]
    else:
        lens = [len(ids) for ids in enc["input_ids"]]

    segments = []
    bucket = [prefix] if prefix else []
    cur_len = pref_len

    for p, tok_len in zip(paragraphs, lens):
        if cur_len + tok_len > max_length:
            if bucket:
                segments.append("\n".join(bucket))
            bucket = ([prefix] if prefix else []) + [p]
            cur_len = pref_len + tok_len
        else:
            bucket.append(p)
            cur_len += tok_len

    if bucket:
        segments.append("\n".join(bucket))

    return segments

def load_prompt(file_path):
    with open(file_path, 'r') as file:
        prompt_template = file.read()
    return prompt_template

def format_prompt(prompt_template, **kwargs):
    return prompt_template.format(**kwargs)

def chunked(it, size):
    it = iter(it)
    return iter(lambda: list(islice(it, size)), [])

def get_new_experiment_path(base_path, prefix="eval_"):

    os.makedirs(base_path, exist_ok=True)
    entries = os.listdir(base_path)
    pattern = re.compile(r'^' + re.escape(prefix) + r'(\d+)$')

    max_id = -1
    for entry in entries:
        match = pattern.match(entry)
        if match:
            exp_id = int(match.group(1))
            if exp_id > max_id:
                max_id = exp_id

    new_id = 0 if max_id == -1 else max_id + 1

    new_path = os.path.join(base_path, f"{prefix}{new_id}")
    os.makedirs(new_path)

    return  new_path

def load_template(prompt_path, method=None, task="description"):
    if method in ["no_context", "truncate"] or method is None:
        return load_prompt(prompt_path)
    elif method=="hierarchical":
        general_prompt_template = load_prompt(os.path.join(prompt_path, "step_0.txt"))
        merge_prompt_template = load_prompt(os.path.join(prompt_path, "merge.txt"))

        return general_prompt_template, merge_prompt_template
    elif method=="incremental":
        general_prompt_template = load_prompt(os.path.join(prompt_path, "first.txt"))
        merge_prompt_template = load_prompt(os.path.join(prompt_path, "prompt.txt"))
        final_path = Path(prompt_path) / "final.txt"
        final_prompt_template = load_prompt(final_path) if final_path.exists() else merge_prompt_template
        return general_prompt_template, merge_prompt_template, final_prompt_template
    raise ValueError(f"Unknown prompt method: {method}")

def parse_think_and_output(text: str):
    """
    Parse a string containing <think>...</think> and output text.

    Args:
        text (str): Input string with <think>...</think> and output.

    Returns:
        dict: {"thinking": str, "output": str}
    """
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)", text, re.DOTALL)
    if not match:
        return {"thinking": None, "output": text.strip()}

    thinking, output = match.groups()
    return {
        "thinking": thinking.strip(),
        "output": output.strip()
    }
