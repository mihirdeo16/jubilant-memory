# Write an introduction to the script
"""
This script is used to train a model using the Huggingface library at scale.
This will run on clusters and use the distributed training capabilities of the Huggingface library.

To run the script, with following command:
python -m torch.distributed.launch --nproc_per_node=4 huggingface_train.py

Following libraries need to be installed:
pip install transformers datasets peft evaluate numpy torch huggingface_hub

"""
import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

import evaluate
from datasets import DatasetDict

# This is task specific imports
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set the seed
seed = 741

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set experiment config
path_to_data = ""
experiment_name = ""
model_checkpoint = ""
output_dir = f"model/lora/{experiment_name}"
logging_dir = f"logs/lora/{experiment_name}"

# Load the data
datasets = DatasetDict.load_from_disk(path_to_data)
id2label = {0: "99214", 1: "99213"}

# Set model config
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
                                                           num_labels=len(id2label.keys()), 
                                                           torch_dtype=torch.float16,
                                                           device_map=device, id2label=id2label, 
                                                           label2id={v: k for k, v in id2label.items()}
)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False, r=4,lora_alpha=32, lora_dropout=0.1,
    target_modules=["query", "key", "value"])
model_lora = get_peft_model(model, peft_config)

# Set the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if torch.cuda.device_count() > 1:
    model_lora.is_parallelizable = True
    model_lora.model_parallel = True


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")

    metrics = {
        "eval/accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "eval/f1": f1.compute(predictions=predictions, references=labels)["f1"],
        "eval/recall": recall.compute(predictions=predictions, references=labels)["recall"],
        "eval/precision": precision.compute(predictions=predictions, references=labels)["precision"],
    }

    return metrics


def preprocess_function(examples):
    text = examples["text"]
    return tokenizer(text, truncation=True, max_length=model.config.max_position_embeddings, padding="max_length")


# Tokenize the data and set the data collator
tokenized_data = datasets.map(preprocess_function)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Set the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=logging_dir,

    # Training hyperparameters
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,

    # Speed up training
    # fp16=True,
    bf16=True,

    # Evaluation strategy
    evaluation_strategy="steps", # "epoch",
    save_strategy="steps", # "epoch",
    metric_for_best_model="eval/f1",

    load_best_model_at_end=True,
    push_to_hub=True,

    seed=seed,

    warmup_steps=5,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    max_steps=1000,
    logging_steps=50,
    optim="paged_adamw_8bit",
    save_steps=50,                
    eval_steps=50,              
    do_eval=True,               
    
)

# Set the trainer
trainer = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()