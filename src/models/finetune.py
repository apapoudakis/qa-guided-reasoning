import argparse
import os
from functools import partial

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_STUDIO_DISABLED", "1")

# Unsloth must load before TRL and other libraries that import Transformers.
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import wandb

from src.utils.misc import load_config, truncate_text, format_prompt, save_config, read_jsonl
from src.utils.train_utils import load_unsloth_model

_cache_texts = {}


def build_prompt(book, character, context, description, prompt_template, tokenizer, truncate_length, thinking, model_type="inst_tuned", qa_pairs=None):
    cache_key = (context, truncate_length)
    if cache_key not in _cache_texts:
        truncated_ctx = truncate_text(context, tokenizer, max_length=truncate_length)
        _cache_texts[cache_key] = truncated_ctx
    else:
        truncated_ctx = _cache_texts[cache_key]

    user_txt = format_prompt(
        prompt_template,
        book=book,
        character=character,
        context=truncated_ctx,
    )

    if qa_pairs:
        thinking_part = f"<think>\n{qa_pairs}\n</think>"
    else:
        thinking_part = ""

    messages = [
        {"role": "user", "content": user_txt},
        {"role": "assistant", "content": thinking_part.strip() + description.strip()},
    ]

    t = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=thinking)

    return t

def format_batch(batch, idx, prompt_template, tokenizer, truncate_length, thinking, model_type, qa_data=None):

    qa_pairs = None
    if qa_data is not None:
        if idx >= len(qa_data):
            raise ValueError("Missing QA record for a training example.")
        if idx < len(qa_data):
            qa_item = qa_data[idx].copy()
            if (qa_item["book"], qa_item["character"]) != (batch["book"], batch["character"]):
                raise ValueError(f"QA record {idx} does not match the training example.")
            qa_pairs = ""
            if qa_item["questions"] and isinstance(qa_item["questions"][0], list):
                qa_item["questions"] = [x for sub in qa_item["questions"] for x in sub]
                qa_item["explanations"] = [x for sub in qa_item["explanations"] for x in sub]
                qa_item["answers"] = [x for sub in qa_item["answers"] for x in sub]
                qa_item["types"] = [x for sub in qa_item["types"] for x in sub]

            fields = [qa_item[field] for field in ("questions", "explanations", "answers", "types")]
            if any(not isinstance(value, str) for field in fields for value in field):
                raise ValueError("SFT requires per-seed QA traces, not combined multi-seed predictions.")
            if len({len(field) for field in fields}) != 1:
                raise ValueError("Training QA fields must have equal lengths.")

            for q, e, a, t in zip(qa_item["questions"], qa_item["explanations"], qa_item["answers"], qa_item["types"]):
                qa_pairs += f"Q: {q}\nExplanation: {e}\nA: {a}\n"
            qa_pairs = f"I need to describe {batch['character']} from the book {batch['book']} based on the given context. Let me start by going through the provided text and answering questions about {batch['character']} one by one.\n\n{qa_pairs}"

    text = build_prompt(batch["book"], batch["character"], batch["input"], batch["description"], prompt_template=prompt_template,
                        tokenizer=tokenizer, truncate_length=truncate_length, thinking=thinking, model_type=model_type, qa_pairs=qa_pairs)


    return {"text": text}

def load_data(data_config):
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_config["data_path"], "train.jsonl"),
        split="train",
        streaming=True,
    )

    val_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_config["data_path"], "val.jsonl"),
        split="train",
        streaming=True,           # <-- enables on-the-fly loading
    )


    return train_dataset, val_dataset

def main(config):
    data_config, train_config, experiment_config, lora_config = config["data_params"], config["train_params"], config["experiment_params"], config["lora_params"]

    model, tokenizer = load_unsloth_model(train_config["model_name"], train_config["max_input_length"], inference=False, full_finetuning=not train_config["apply_lora"])


    if train_config["apply_lora"]:
        model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],  # 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth", # "unsloth",  # True or "unsloth" for very long context
        random_state=train_config["seed"],
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        )

    if data_config["qa_data_path"] is not None:
        qa_data_train = read_jsonl(os.path.join(data_config["qa_data_path"], "train.jsonl"))
        qa_data_val = read_jsonl(os.path.join(data_config["qa_data_path"], "val.jsonl"))
    else:
        qa_data_train, qa_data_val = None, None
    experiment_path = os.path.join(experiment_config['save_experiment'], experiment_config['model'], experiment_config['experiment_name'], f"seed_{train_config['seed']}")

    os.makedirs(experiment_path, exist_ok=True)
    if experiment_config.get("report_to") == "wandb":
        wandb.init(
            project=experiment_config["wandb_project"],
            dir=experiment_path,
            name=experiment_config["model"] + "-" + experiment_config["experiment_name"],
            config=config,
        )

    save_config(config, os.path.join(experiment_path, "train.yaml"))

    prompt_template = open(data_config['prompt_path']).read()


    train_data, val_data = load_data(data_config)

    model_type = "inst_tuned" if "inst" in train_config["model_name"].lower() else "base"

    train_format_fn = partial(
        format_batch,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        truncate_length=train_config["truncate_length"],
        thinking = train_config["thinking"],
        model_type=model_type,
        qa_data=qa_data_train,
    )

    val_format_fn = partial(
        format_batch,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        truncate_length=train_config["truncate_length"],
        thinking = train_config["thinking"],
        model_type=model_type,
        qa_data=qa_data_val,
    )

    train_dataset = train_data.map(
        train_format_fn,
        remove_columns=train_data.column_names,
        with_indices=True,
    )
    val_dataset = val_data.map(
        val_format_fn,
        remove_columns=val_data.column_names,
        with_indices=True
    )


    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            packing=False,
            dataset_num_proc=1,
            max_length=train_config["max_input_length"],
            per_device_train_batch_size=train_config["batch_size"],
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
            eval_strategy="steps",
            eval_steps=train_config["eval_steps"],
            save_steps=train_config["save_steps"],
            max_steps=train_config["max_steps"],
            warmup_steps=train_config["warm_up_steps"],
            learning_rate=train_config["lr"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=train_config["logging_steps"],
            optim="adamw_8bit",
            weight_decay=train_config["weight_decay"],
            lr_scheduler_type=train_config["lr_schedule"],
            seed=train_config["seed"],
            output_dir=experiment_path,
            do_eval=True,
            report_to=experiment_config.get("report_to", "none"),
            save_strategy=train_config["save_strategy"],
            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",
            label_smoothing_factor = train_config["label_smoothing"],
            ignore_data_skip=True
        ),
    )

    if "mistral" in train_config["model_name"].lower():
        trainer = train_on_responses_only(trainer,
            instruction_part = "[INST]",
            response_part = "[/INST]",
        )
    else:
        trainer = train_on_responses_only(trainer,
            instruction_part = "<|im_start|>user",
            response_part = "<|im_start|>assistant",
        )


    trainer.train(resume_from_checkpoint=train_config["resume_from_checkpoint"])

    if train_config.get("save_vllm_format", True):
        model.save_pretrained_merged(os.path.join(experiment_path, "best_vllm_format_model"), tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-d', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(load_config(args.config))
