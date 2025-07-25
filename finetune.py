import os
import json
import torch
import numpy as np
import platform
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import logging

# --- Setup ---
# Configure logging for clear and informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run a fine-tuning process with standard, well-researched
    hyperparameters for LayoutLMv3 on a powerful local machine.
    """
    # --- 1. Define Paths and Load Data ---
    processed_dataset_path = "./processed_doclaynet"
    labels_file_path = os.path.join(processed_dataset_path, "labels.json")

    if not os.path.exists(processed_dataset_path) or not os.path.exists(labels_file_path):
        raise FileNotFoundError(
            f"Ensure both '{processed_dataset_path}' directory and '{labels_file_path}' exist. "
            "Run the data creation script first."
        )

    logger.info(f"Loading pre-processed dataset from {processed_dataset_path}...")
    processed_dataset = load_from_disk(processed_dataset_path)
    processed_dataset.set_format("torch")
    
    logger.info(f"Loading label mappings from {labels_file_path}...")
    with open(labels_file_path, 'r') as f:
        label_list = json.load(f)
    
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    logger.info("✅ Data and labels loaded successfully!")
    
    # --- 2. Define Evaluation Metric ---
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = predictions[labels != -100]
        true_labels = labels[labels != -100]
        return {"accuracy": (true_predictions == true_labels).mean()}

    # --- 3. Load Model and Processor ---
    logger.info("Loading LayoutLMv3 model and processor...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        id2label=id2label,
        label2id=label2id
    )
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    
    # --- 4. Define Standard & Refined Training Arguments ---
    logger.info("Configuring training arguments with standard best practices...")
    use_torch_compile = hasattr(torch, '__version__') and torch.__version__.startswith("2") and platform.system() != "Windows"

    training_args = TrainingArguments(
        output_dir="./layoutlmv3-finetuned-standard",
        num_train_epochs=5,
        # Standard Hyperparameters:
        learning_rate=1e-5,                  # Lower learning rate for stable fine-tuning.
        weight_decay=0.01,                   # Standard weight decay for regularization.
        lr_scheduler_type="cosine",          # Cosine decay is a modern standard.
        warmup_steps=500,                    # Standard fixed number of warmup steps.
        # Performance Optimizations for your RTX A5000:
        per_device_train_batch_size=8,       # Reduced slightly for stability, still high.
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,       # Maintains a large effective batch size of 32.
        bf16=torch.cuda.is_bf16_supported(), # Best performance on Ampere GPUs.
        dataloader_num_workers=max(0, os.cpu_count() - 2),
        torch_compile=use_torch_compile,
        # Logging and Saving:
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
    )

    # --- 5. Initialize the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    # --- 6. Start Training! ---
    logger.info("\n" + "="*60)
    logger.info("🚀 Starting fine-tuning with standard hyperparameters! 🚀")
    if use_torch_compile: logger.info("✅ `torch.compile` is ENABLED.")
    if training_args.bf16: logger.info("✅ `bfloat16` is ENABLED.")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info("="*60 + "\n")
    
    trainer.train()

    # --- 7. Save the Final Model and Processor ---
    logger.info("\nTraining finished. Saving the best model and processor...")
    final_model_path = './my-final-doclaynet-model-local'
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    logger.info(f"\n✅ Model and processor saved successfully to '{final_model_path}'")

if __name__ == "__main__":
    main()