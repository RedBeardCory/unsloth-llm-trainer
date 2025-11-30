"""
Training script for fine-tuning LLaMA models on code quality advice using Unsloth.

This script:
1. Loads a base model (e.g., Llama 3.2)
2. Loads training data from training_data.jsonl
3. Fine-tunes the model using LoRA
4. Saves the model in GGUF format for Ollama
5. Creates an Ollama-compatible model

Usage:
    python train_model.py
"""

import os

import torch
import unsloth
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, apply_chat_template, is_bfloat16_supported

# Disable PyTorch compilation features that require C++ compiler on Windows
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Configuration
MAX_SEQ_LENGTH = 2048  # Adjust based on your data
MODEL_NAME = "codellama/CodeLlama-7b-hf"  # Official CodeLlama from HuggingFace
LORA_RANK = 8
LORA_ALPHA = 32
OUTPUT_DIR = "outputs"
TRAINING_DATA_FILE = "training_data.jsonl"
LEARNING_RATE: float = 1e-5
NUM_TRAIN_EPOCHS = 1

CHAT_TEMPLATE = """Below are some interactions between a user and an assistant.

### User:
{INPUT}

### Assistant:
{OUTPUT}"""


def load_model():
    """Load the base model with 4-bit quantization for efficiency."""
    print(f"Loading model: {MODEL_NAME}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization to save memory
    )

    # Configure LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,  # Optimized for speed
        bias="none",
        use_gradient_checkpointing="unsloth",  # Long context support
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def load_training_data(tokenizer):
    """Load and format the training data."""
    print(f"Loading training data from {TRAINING_DATA_FILE}")

    # Load dataset from JSONL file
    dataset = load_dataset("json", data_files=TRAINING_DATA_FILE, split="train")

    print(f"Loaded {len(dataset)} training examples")

    # TODO: just alter the training data generation function so we don't have to do this
    dataset = dataset.rename_column("messages", "conversations")

    chat_template = """Below are some interactions between a user and an assistant.

### User:
{INPUT}

### Assistant:
{OUTPUT}"""

    dataset = apply_chat_template(
        dataset,
        tokenizer=tokenizer,
        chat_template=chat_template,
        default_system_message="You are an expert code reviewer. Analyze code for quality, best practices, and potential improvements.",
    )

    return dataset


def train_model(model, tokenizer, dataset):
    """Fine-tune the model on the training data."""
    print("Starting training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=NUM_TRAIN_EPOCHS,  # Adjust based on your data size
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none",  # Change to "wandb" if you want logging
        ),
    )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    return trainer_stats


def save_for_ollama(model, tokenizer):
    """Save the model in GGUF format and create Ollama model."""
    print("\nSaving model for Ollama...")

    # Save LoRA adapters first (optional, for backup)
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    print("Saved LoRA adapters to lora_model/")

    # Export to GGUF format
    # q8_0 = highest quality, ~8GB
    # q4_k_m = good balance, ~4GB (recommended)
    # f16 = full precision, largest file
    print("Exporting to GGUF format (this may take a few minutes)...")

    model.save_pretrained_gguf(
        "model_gguf",
        tokenizer,
        quantization_method="q4_k_m",  # Good balance of quality and size
    )

    print("\nModel saved! To use with Ollama:")
    print("1. Make sure Ollama is installed and running:")
    print("   curl -fsSL https://ollama.ai/install.sh | sh")
    print("   ollama serve")
    print()
    print("2. Create the Ollama model:")
    print("   ollama create code-advisor -f Modelfile")
    print()
    print("3. Test the model:")
    print('   ollama run code-advisor "Review this Python code..."')


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Code Quality Advisor - Model Training")
    print("=" * 60)

    # Step 1: Load model
    model, tokenizer = load_model()

    # Step 2: Load training data
    dataset = load_training_data(tokenizer)

    # Step 3: Train model
    train_model(model, tokenizer, dataset)

    # Step 4: Save for Ollama
    save_for_ollama(model, tokenizer)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
