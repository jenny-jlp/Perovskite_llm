import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

def train():
    # 1. Configuration
    model_id = "Qwen/Qwen2.5-3B-Instruct"  # You can also change this to "Qwen/Qwen2.5-7B-Instruct"
    train_data_path = "./data/processed/perovskite_llm_train.jsonl"
    test_data_path = "./data/processed/perovskite_llm_test.jsonl"
    output_dir = "./output/perovskite_qwen_lora"
    smoke_test = os.environ.get("TRAIN_LORA_SMOKE_TEST") == "1"
    max_steps_override = int(os.environ.get("TRAIN_LORA_MAX_STEPS", "-1"))
    skip_final_save = os.environ.get("TRAIN_LORA_SKIP_FINAL_SAVE") == "1"
    if smoke_test:
        output_dir = "./output/_smoke_train_lora"
        print("Running TRAIN_LORA_SMOKE_TEST=1: using tiny data slices and one training step.")
    
    print(f"CUDA status: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("WARNING: CUDA is not available. Training will be extremely slow on CPU!")
        return

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"BF16 supported: {bf16_supported}")

    last_checkpoint = None
    resume_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if name.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, name))
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda path: int(path.rsplit("-", 1)[-1]))
            print(f"Found checkpoint: {last_checkpoint}")
            resume_checkpoint = last_checkpoint

    # Use fp16 when resuming from a checkpoint that has scaler.pt (was saved in fp16 mode).
    use_bf16 = bf16_supported and not (
        last_checkpoint and os.path.exists(os.path.join(last_checkpoint, "scaler.pt"))
    )
    if bf16_supported and not use_bf16:
        print("Using fp16 for checkpoint compatibility. Start a fresh run to use bf16.")

    # 2. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load and Format Dataset
    print("Loading datasets...")
    dataset = load_dataset("json", data_files={
        "train": train_data_path,
        "test": test_data_path
    })
    if smoke_test:
        dataset["train"] = dataset["train"].select(range(8))
        dataset["test"] = dataset["test"].select(range(4))

    # Qwen chat template formatting function
    def format_dataset(example):
        # We format as a chat conversation: System -> User (Instruction) -> Assistant (Output)
        messages = [
            {"role": "system", "content": "You are a materials science AI assistant specialized in perovskite solar cells."},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        
        # Apply the model's standard chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Tokenize the complete text
        tokenized = tokenizer(text, truncation=True, max_length=1024, padding=False)
        
        # Create labels for causal masking (we only want to compute loss on the assistant's response)
        # Find where the assistant response starts in the text
        user_prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        user_prompt_len = len(tokenizer(user_prompt_text, truncation=True, max_length=1024)["input_ids"])
        
        labels = [-100] * len(tokenized["input_ids"])
        # Only compute loss on the generated assistant tokens (everything after the prompt length)
        for i in range(user_prompt_len, len(tokenized["input_ids"])):
            labels[i] = tokenized["input_ids"][i]
            
        tokenized["labels"] = labels
        return tokenized

    print("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        format_dataset,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset"
    )

    # 4. Quantization Configuration (QLoRA)
    # Using 4-bit quantization to fit easily within 16GB VRAM with plenty of headroom
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 5. Load Base Model
    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for 8/4-bit training (gradients checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)

    # 6. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    print("\n--- Trainable Parameters ---")
    model.print_trainable_parameters()

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1 if smoke_test else 4,
        per_device_eval_batch_size=1 if smoke_test else 4,
        gradient_accumulation_steps=1 if smoke_test else 4,  # Effective batch size = 4 * 4 = 16
        learning_rate=2e-4,
        logging_steps=1 if smoke_test else 10,
        eval_strategy="no" if smoke_test else "steps",
        eval_steps=500,
        save_strategy="no" if smoke_test else "steps",
        save_steps=500,
        num_train_epochs=3,
        max_steps=1 if smoke_test else max_steps_override,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch",   # Changed from paged_adamw_8bit to avoid bitsandbytes KeyError
        gradient_checkpointing=True,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",  # Change to "tensorboard" or "wandb" if desired
        save_total_limit=3,
        load_best_model_at_end=not smoke_test,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # 8. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )

    # Disable cache to save memory and avoid training warnings
    model.config.use_cache = False

    # 9. Start Training
    print("Starting fine-tuning training...")
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 10. Save trained model weights
    if skip_final_save:
        print("Skipping final LoRA adapter save because TRAIN_LORA_SKIP_FINAL_SAVE=1.")
    else:
        print(f"Saving final LoRA adapter to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Training finished and LoRA model saved successfully!")

if __name__ == "__main__":
    train()
