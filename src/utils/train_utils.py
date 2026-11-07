def load_unsloth_model(checkpoint_path, max_input_length, inference=True, full_finetuning=False):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=max_input_length,
        dtype=None,
        load_in_4bit=False,
        full_finetuning=full_finetuning,
        device_map="auto"

    )
    tokenizer.model_max_length = max_input_length
    if inference:
        FastLanguageModel.for_inference(model)

    print("model_max_length =", tokenizer.model_max_length)

    return model, tokenizer

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def convert_chkpt(hf_dir, out_dir):
    model, tokenizer = load_unsloth_model(hf_dir, 32768)

    model.save_pretrained_merged(out_dir, tokenizer, save_method = "merged_16bit",)
