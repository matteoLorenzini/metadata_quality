import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import os

# --- Configuration ---
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/CH_IT/active_learning_label/dataset_active_learning.csv'))
MODEL_DIR = "./iccd_bert_active_model"
TEXT_COLUMN = 'descrizione'
LABEL_COLUMN = 'label'
model_name = "dbmdz/bert-base-italian-cased"

# --- Load all labeled data ---
df = pd.read_csv(DATA_PATH)
label2id = {label: idx for idx, label in enumerate(sorted(df[LABEL_COLUMN].unique()))}
id2label = {v: k for k, v in label2id.items()}
df['label_id'] = df[LABEL_COLUMN].map(label2id)

# --- Tokenizer and dataset preparation ---
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        padding='max_length',
        max_length=128,
    )

dataset = Dataset.from_pandas(df[[TEXT_COLUMN, 'label_id']])
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label_id", "labels")
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# --- Model ---
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# --- Training arguments (use your best found parameters) ---
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,                # Example: your best epoch count
    learning_rate=3e-5,                # Example: your best learning rate
    per_device_train_batch_size=16,    # Example: your best batch size
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# --- Save model and tokenizer ---
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Model and tokenizer saved to '{MODEL_DIR}'")