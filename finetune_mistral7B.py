%cd /content/drive/MyDrive/depin

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback  # Add this import
)
import os
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from datetime import datetime
from torch.cuda import empty_cache
from tqdm import tqdm
import json
from collections import Counter

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET_PATH = "depin_exploits1.txt"
DATA_SPLIT = 0.1
MAX_INPUT_TOKENS = 480

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 3  # Number of epochs to wait for improvement
EARLY_STOPPING_THRESHOLD = 0.001  # Minimum improvement threshold

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"./mistral-exploit-classifier_{timestamp}"

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and split dataset
dataset = load_dataset("json", data_files=DATASET_PATH)["train"]
dataset = dataset.train_test_split(test_size=DATA_SPLIT, shuffle=True, seed=42)

# Save the split datasets to OUTPUT_DIR
train_file = os.path.join(OUTPUT_DIR, f"train_data_{timestamp}.json")
val_file = os.path.join(OUTPUT_DIR, f"val_data_{timestamp}.json")

dataset["train"].to_json(train_file)
dataset["test"].to_json(val_file)

print(f"Training data saved to {train_file}")
print(f"Validation data saved to {val_file}")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)

# Disable gradient checkpointing for 4-bit models
model.config.use_cache = True  # Required for 4-bit models

def tokenize_function(examples):
    processed_texts = []
    for inp, out in zip(examples['input'], examples['output']):
        # Tokenize input separately to check length
        input_tokens = tokenizer.encode(inp, add_special_tokens=False)
        if len(input_tokens) > MAX_INPUT_TOKENS:
            # Truncate input tokens if too long
            input_tokens = input_tokens[:MAX_INPUT_TOKENS]
            inp = tokenizer.decode(input_tokens, skip_special_tokens=True)

        # Construct the full text with prompt
        text = f"<s>Input: {inp}</s> Output: [[[{out}]]]"
        processed_texts.append(text)

    tokenized = tokenizer(
        processed_texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
        return_attention_mask=True
    )
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

def compute_metrics(eval_preds):
    model.eval()
    predictions = []
    labels = []

    for example in tqdm(dataset["test"], position=0):
        try:
            prompt = f"<s>Input: {example['input']}</s> Output:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():  # Disable gradients for validation
                output = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            pred_exploit = re.search(r'\[\[\[(.*?)\]\]\]', generated_text)
            pred_exploit = pred_exploit.group(1).strip() if pred_exploit else "None"
            true_exploit = example['output']

            predictions.append(pred_exploit)
            labels.append(true_exploit)

            del inputs, output
            empty_cache()

        except Exception as e:
            print(f"Error processing example: {e}")
            continue

    # Get all unique classes in the dataset
    unique_classes = sorted(set(labels))
    counts = dict(Counter(labels))

    # Calculate per-class metrics
    precision = precision_score(labels, predictions, average=None, labels=unique_classes, zero_division=0)
    recall = recall_score(labels, predictions, average=None, labels=unique_classes, zero_division=0)
    f1 = f1_score(labels, predictions, average=None, labels=unique_classes, zero_division=0)

    # Create a dictionary of per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(unique_classes):
        per_class_metrics[f"count_{class_name}"] = counts[class_name]
        per_class_metrics[f"precision_{class_name}"] = float(precision[i])
        per_class_metrics[f"recall_{class_name}"] = float(recall[i])
        per_class_metrics[f"f1_{class_name}"] = float(f1[i])

    # Add macro-averaged metrics
    macro_metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0)
    }
    # Combine all metrics
    metrics = {**macro_metrics, **per_class_metrics}


    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "a") as f:
        json.dump(metrics, f, indent=4)
        f.write(",")


    return metrics

# Updated TrainingArguments with early stopping configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    eval_accumulation_steps=1,
    fp16_full_eval=True,
    num_train_epochs=100,  # Set high, early stopping will handle termination
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",  # Must evaluate every epoch for early stopping
    learning_rate=2e-5,
    fp16=True,
    warmup_steps=10,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,  # F1 score should be maximized
    optim="paged_adamw_8bit",
    save_total_limit=3  # Keep only the best 3 checkpoints to save space
)

# Create early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=EARLY_STOPPING_THRESHOLD
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]  # Add the early stopping callback
)

print("Starting training with early stopping...")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
print(f"Early stopping threshold: {EARLY_STOPPING_THRESHOLD}")

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# Print final training summary
print("\nTraining completed!")
print(f"Total epochs trained: {trainer.state.epoch}")
if trainer.state.epoch < training_args.num_train_epochs:
    print("Training stopped early due to lack of improvement in F1 score.")
else:
    print("Training completed all epochs.")
