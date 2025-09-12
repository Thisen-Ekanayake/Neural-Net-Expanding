import os
import time
import builtins
from datetime import datetime

import torch
import wandb
import sentencepiece as spm
from transformers import (
    GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
)
from collator import CustomDataCollator

from dataset import SinhalaTokenDataset
from logger import Logger


# --- Setup Logger ---
# Modified logger to handle file parameter
class CustomPrint:
    def __init__(self, logger):
        self.logger = logger
    
    def __call__(self, *args, **kwargs):
        if 'file' in kwargs:
            del kwargs['file']  # Remove file parameter if present
        message = ' '.join(str(arg) for arg in args)
        self.logger.write(message, **kwargs)

log = Logger(log_dir="logs/train_run")
builtins.print = CustomPrint(log)  # Use our custom print handler


# --- Load SentencePiece Tokenizer ---
sp = spm.SentencePieceProcessor()
sp.Load("")  


# --- Custom Trainer with Logging ---
class TimingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_start_time = None
        self.step_start_time = None

    def training_step(self, model, inputs, num_items_in_batch=None):  # Added num_items_in_batch
        if self.step_start_time is None:
            self.step_start_time = time.time()
        output = super().training_step(model, inputs)
        step_time = time.time() - self.step_start_time
        print(f"[Step {self.state.global_step}] Training step took {step_time:.3f}s")
        self.step_start_time = time.time()
        return output

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"=== Epoch {state.epoch:.2f} started ===")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        print(f"=== Epoch {state.epoch:.2f} ended, time taken: {epoch_time:.2f}s ===")


# --- Main Training Script ---
def main():
    start_time = time.time()
    print(f"Script started at {datetime.now()}")

    model_config_path = "model-config.json"
    data_file = " "
    run_name = "sinhala-gpt-v1"

    # Load model config and dataset
    config = GPT2Config.from_json_file(model_config_path)
    dataset = SinhalaTokenDataset(data_file)
    print(f"Loaded model config and dataset with {len(dataset)} samples")

    model = GPT2LMHeadModel(config)
    data_collator = CustomDataCollator(pad_token_id=0)  # 0 is <pad> in your tokenizer

    # Updated TrainingArguments for manual eval + save after each epoch
    training_args = TrainingArguments(
        output_dir="checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=6,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_dir="logs",
        logging_steps=500,
        save_strategy="epoch",  # Save model after every epoch
        save_total_limit=3,
        learning_rate=5e-4,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        fp16=True,
        report_to=["wandb"],
        run_name=run_name,
        seed=42,
    )

    # Initialize wandb
    wandb.init(project="SinhalaGPT", name=run_name, config=training_args.to_dict())

    trainer = TimingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("final-model")
    wandb.finish()

    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes")
    log.close()


if __name__ == "__main__":
    main()