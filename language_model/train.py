import os
import time
import builtins
import json
import h5py
import numpy as np
import uuid
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
class CustomPrint:
    def __init__(self, logger):
        self.logger = logger
    
    def __call__(self, *args, **kwargs):
        # Handle the 'end' parameter by converting it to part of the message
        end_char = kwargs.get('end', '\n')
        message = ' '.join(str(arg) for arg in args) + end_char
        self.logger.write(message)

log = Logger(log_dir="logs/train_run")
builtins.print = CustomPrint(log)


# --- Load SentencePiece Tokenizer ---
sp = spm.SentencePieceProcessor()
sp.Load("../unigram_4000_0.9995.model")  


# --- Parameter Logging Callback ---
class ParameterLogger:
    def __init__(self, log_dir="parameter_logs", log_interval=100):
        self.log_dir = log_dir
        self.log_interval = log_interval
        os.makedirs(log_dir, exist_ok=True)
        
        # Create HDF5 file for detailed parameter storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.h5_file_path = os.path.join(log_dir, f"parameters_{timestamp}.h5")
        self.h5_file = h5py.File(self.h5_file_path, "w")
        self.step_count = 0
        self.logged_steps = set()  # Track which steps we've already logged
        
    def log_parameters(self, model, step, epoch, is_gradient=False, phase=""):
        """Log all parameters and their statistics"""
        # Create a unique identifier for this logging event
        unique_id = str(uuid.uuid4())[:8]
        prefix = "gradients/" if is_gradient else "parameters/"
        group_name = f"{prefix}step_{step}_epoch_{epoch:.2f}_{phase}_{unique_id}"
        
        # Check if we've already logged this step to avoid duplicates
        if step in self.logged_steps and is_gradient:
            return  # Skip if we've already logged gradients for this step
        
        param_group = self.h5_file.create_group(group_name)
        
        # Store parameter metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "is_gradient": is_gradient,
            "phase": phase
        }
        param_group.attrs.update(metadata)
        
        # Log each parameter
        for name, param in model.named_parameters():
            if param.requires_grad and (is_gradient or param.grad is not None):
                # Clean parameter name for HDF5
                clean_name = name.replace('.', '_').replace(' ', '_')
                
                # Get parameter data
                if is_gradient and param.grad is not None:
                    data = param.grad.detach().cpu().numpy()
                else:
                    data = param.detach().cpu().numpy()
                
                # Store parameter data
                param_group.create_dataset(clean_name, data=data)
                
                # Calculate and store statistics
                stats = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "shape": data.shape
                }
                
                # Add statistics as attributes
                param_ds = param_group[clean_name]
                param_ds.attrs.update(stats)
                
                # Also log to wandb for visualization
                if self.step_count % self.log_interval == 0:
                    wandb.log({
                        f"{'gradient_' if is_gradient else ''}{name}/mean": stats["mean"],
                        f"{'gradient_' if is_gradient else ''}{name}/std": stats["std"],
                        f"{'gradient_' if is_gradient else ''}{name}/min": stats["min"],
                        f"{'gradient_' if is_gradient else ''}{name}/max": stats["max"],
                    }, step=step)
        
        # Mark this step as logged for gradients
        if is_gradient:
            self.logged_steps.add(step)
            
        # Log layer-wise statistics
        self.log_layer_statistics(model, step, epoch, is_gradient)
    
    def log_layer_statistics(self, model, step, epoch, is_gradient=False):
        """Log statistics for each layer"""
        prefix = "gradient_" if is_gradient else ""
        layer_stats = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                layer_params = []
                for param_name, param in module.named_parameters(recurse=False):
                    if is_gradient and param.grad is not None:
                        data = param.grad.detach().cpu().numpy()
                    elif not is_gradient:
                        data = param.detach().cpu().numpy()
                    else:
                        continue
                    
                    layer_params.append(data)
                
                if layer_params:
                    all_params = np.concatenate([p.flatten() for p in layer_params])
                    layer_stats[f"{prefix}{name}/mean"] = float(np.mean(all_params))
                    layer_stats[f"{prefix}{name}/std"] = float(np.std(all_params))
                    layer_stats[f"{prefix}{name}/min"] = float(np.min(all_params))
                    layer_stats[f"{prefix}{name}/max"] = float(np.max(all_params))
        
        # Log to wandb
        if layer_stats and self.step_count % self.log_interval == 0:
            wandb.log(layer_stats, step=step)
    
    def close(self):
        self.h5_file.close()


# --- Custom Trainer with Enhanced Logging ---
class TimingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_start_time = None
        self.step_start_time = None
        self.param_logger = ParameterLogger(log_interval=100)
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.step_start_time is None:
            self.step_start_time = time.time()
        
        # Forward pass
        outputs = super().training_step(model, inputs, num_items_in_batch)
        
        # Log gradients after backward pass
        if self.state.global_step % self.param_logger.log_interval == 0:
            self.param_logger.log_parameters(
                model, self.state.global_step, 
                self.state.epoch, is_gradient=True, phase="training_step"
            )
        
        step_time = time.time() - self.step_start_time
        print(f"[Step {self.state.global_step}] Training step took {step_time:.3f}s")
        self.step_start_time = time.time()
        
        return outputs

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"=== Epoch {state.epoch:.2f} started ===")
        
        # Log parameters at the beginning of each epoch
        self.param_logger.log_parameters(
            self.model, state.global_step, state.epoch, 
            phase="epoch_begin"
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        print(f"=== Epoch {state.epoch:.2f} ended, time taken: {epoch_time:.2f}s ===")
        
        # Log parameters at the end of each epoch
        self.param_logger.log_parameters(
            self.model, state.global_step, state.epoch, 
            phase="epoch_end"
        )
        
        # Save model checkpoint with full state
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{state.epoch:.2f}")
        self.save_model(checkpoint_path)
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        
        # Save learning rate scheduler state
        torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))
        
        # Save training state
        torch.save(self.state, os.path.join(checkpoint_path, "trainer_state.pt"))

    def on_train_end(self, args, state, control, **kwargs):
        # Close the parameter logger
        self.param_logger.close()
        
        # Save final model with all parameters
        super().on_train_end(args, state, control, **kwargs)


# --- Additional Metrics Tracking ---
def add_parameter_metrics_to_log(log_dict, model, prefix=""):
    """Add parameter metrics to the log dictionary"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Parameter values
            data = param.detach().cpu().numpy()
            log_dict[f"{prefix}{name}/mean"] = float(np.mean(data))
            log_dict[f"{prefix}{name}/std"] = float(np.std(data))
            
            # Gradients if available
            if param.grad is not None:
                grad_data = param.grad.detach().cpu().numpy()
                log_dict[f"{prefix}{name}/grad_mean"] = float(np.mean(grad_data))
                log_dict[f"{prefix}{name}/grad_std"] = float(np.std(grad_data))
                log_dict[f"{prefix}{name}/grad_norm"] = float(torch.norm(param.grad).item())


# --- Main Training Script ---
def main():
    start_time = time.time()
    print(f"Script started at {datetime.now()}")

    model_config_path = "model-config.json"
    data_file = "../100M_dataset.jsonl"
    run_name = "sinhala-gpt-v1"

    # Load model config and dataset
    config = GPT2Config.from_json_file(model_config_path)
    dataset = SinhalaTokenDataset(data_file)
    print(f"Loaded model config and dataset with {len(dataset)} samples")

    model = GPT2LMHeadModel(config)
    data_collator = CustomDataCollator(pad_token_id=0)

    # Updated TrainingArguments
    training_args = TrainingArguments(
        output_dir="checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=6,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_dir="logs",
        logging_steps=500,
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=5e-4,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        fp16=True,
        report_to=["wandb"],
        run_name=run_name,
        seed=42,
        # Additional settings for better logging
        logging_first_step=True,
        logging_nan_inf_filter=False,  # Don't filter out nan/inf values
    )

    # Initialize wandb with more detailed configuration
    wandb.init(
        project="SinhalaGPT", 
        name=run_name, 
        config={
            **training_args.to_dict(),
            "model_config": config.to_dict(),
            "dataset_size": len(dataset)
        }
    )

    trainer = TimingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Store the original log method
    original_log = trainer.log

    # Create a wrapper that accepts both arguments
    def extended_log(logs, start_time=None):
        # Add parameter metrics to the log
        add_parameter_metrics_to_log(logs, model)
        # Call the original log method with both arguments
        return original_log(logs, start_time)

    # Replace the log method
    trainer.log = extended_log

    trainer.train()
    trainer.save_model("final-model")
    
    # Save final parameter state
    final_param_logger = ParameterLogger(log_dir="final_parameters")
    final_param_logger.log_parameters(
        model, trainer.state.global_step, trainer.state.epoch,
        phase="final"
    )
    final_param_logger.log_parameters(
        model, trainer.state.global_step, trainer.state.epoch, 
        is_gradient=True, phase="final_grad"
    )
    final_param_logger.close()
    
    wandb.finish()

    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes")
    log.close()


if __name__ == "__main__":
    main()