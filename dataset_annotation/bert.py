import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os
import transformers
from transformers import EarlyStoppingCallback

# --- Configuration ---
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/CH_IT/active_learning_label/dataset_active_learning.csv'))
UNLABELED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/CH_IT/active_learning_label/unlabelled_batches/unlabelled_batch_3.csv'))
ANNOTATION_BATCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/CH_IT/active_learning_label/annotation_batches/annotation_batch_1.csv'))
LABELLED_WITH_PRED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/CH_IT/active_learning_label/labelled_batches/labelled_with_predictions.csv'))

TEXT_COLUMN = 'descrizione'
LABEL_COLUMN = 'label'
model_name = "dbmdz/bert-base-italian-cased"

print("Transformers version:", transformers.__version__)
from transformers import TrainingArguments
print("TrainingArguments location:", TrainingArguments.__module__)
print("TrainingArguments file:", TrainingArguments.__init__.__code__.co_filename)

# --- Load labeled data ---
df_labeled = pd.read_csv(DATA_PATH)
label2id = {label: idx for idx, label in enumerate(sorted(df_labeled[LABEL_COLUMN].unique()))}
id2label = {v: k for k, v in label2id.items()}
df_labeled['label_id'] = df_labeled[LABEL_COLUMN].map(label2id)

# --- Train/val split ---
train_df, val_df = train_test_split(
    df_labeled,
    stratify=df_labeled['label_id'],
    test_size=0.2,
    random_state=42
)

# --- Tokenizer and dataset preparation ---
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        padding='max_length',
        max_length=128,
    )

def prepare_dataset(df, labeled=True):
    columns = [TEXT_COLUMN, 'label_id'] if labeled else [TEXT_COLUMN]
    dataset = Dataset.from_pandas(df[columns])
    dataset = dataset.map(tokenize, batched=True)
    if labeled:
        # Rename 'label_id' to 'labels' for the Trainer
        dataset = dataset.rename_column("label_id", "labels")
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels']) # Updated here
    else:
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

train_dataset = prepare_dataset(train_df, labeled=True)
val_dataset = prepare_dataset(val_df, labeled=True)

# --- Model ---
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# --- Training arguments with fine-tuning suggestions ---
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,                # Increased epochs for better convergence
    learning_rate=3e-5,                # Lower learning rate for finer updates
    per_device_train_batch_size=8,    # Larger batch size (if GPU allows)
    per_device_eval_batch_size=8,
    weight_decay=0.01,                 # Regularization to prevent overfitting
    eval_strategy="epoch",
    save_strategy="epoch",                # <-- add this line
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,              # <-- add this
    metric_for_best_model="eval_loss",        # <-- and this (or another metric)
)

# --- Trainer with early stopping ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Early stopping if no improvement for 2 evals
)

trainer.train()

# --- Save the trained model ---
trainer.save_model('./iccd_bert_active_model')
tokenizer.save_pretrained('./iccd_bert_active_model')
print("Model and tokenizer saved to './iccd_bert_active_model'")

# --- Training performance on all labeled data ---
full_dataset = prepare_dataset(df_labeled, labeled=True)
train_outputs = trainer.predict(full_dataset)
train_preds = train_outputs.predictions.argmax(axis=1)
train_true = df_labeled['label_id'].values
print("\nClassification report on training data:")
print(classification_report(train_true, train_preds, target_names=[id2label[i] for i in range(len(id2label))]))

# --- Validation performance ---
val_outputs = trainer.predict(val_dataset)
val_preds = val_outputs.predictions.argmax(axis=1)
val_true = val_df['label_id'].values
print("\nClassification report on validation data:")
print(classification_report(val_true, val_preds, target_names=[id2label[i] for i in range(len(id2label))]))

# --- Predict on unlabeled dataset for active learning ---
df_unlabeled = pd.read_csv(UNLABELED_PATH)
unlabeled_dataset = prepare_dataset(df_unlabeled, labeled=False)

unlabeled_outputs = trainer.predict(unlabeled_dataset)
unlabeled_probs = torch.nn.functional.softmax(torch.tensor(unlabeled_outputs.predictions), dim=1).numpy()
unlabeled_preds = unlabeled_probs.argmax(axis=1)
unlabeled_confidence = unlabeled_probs.max(axis=1)

df_unlabeled['predicted_label_id'] = unlabeled_preds
df_unlabeled['predicted_class'] = [id2label[x] for x in unlabeled_preds]
df_unlabeled['confidence'] = unlabeled_confidence
for i, class_name in id2label.items():
    df_unlabeled[f'prob_{class_name}'] = unlabeled_probs[:, i]

# Save all predictions
df_unlabeled.to_csv(LABELLED_WITH_PRED_PATH, index=False)
print(f"\nPredictions on unlabeled data saved to '{LABELLED_WITH_PRED_PATH}'")

# --- Active Learning: Select most uncertain samples for annotation ---
N = 20  # Number of samples to select for annotation
df_unlabeled['uncertainty'] = 1 - df_unlabeled['confidence']
to_annotate = df_unlabeled.sort_values('uncertainty', ascending=False).head(N)
to_annotate[[TEXT_COLUMN, 'predicted_class', 'confidence', 'uncertainty']].to_csv(
    ANNOTATION_BATCH_PATH, index=False
)
print(f"\nTop {N} most uncertain samples saved to '{ANNOTATION_BATCH_PATH}' for annotation.")