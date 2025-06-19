import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_DIR = "./iccd_bert_active_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
)

input_files = [
    "../../../dataset/CH_IT/architettura/merged_unlabelled_architettura.csv",
    "../../../dataset/CH_IT/archeologia/merged_unlabelled_archeologia.csv",
    "../../../dataset/CH_IT/opere_arte_visiva/merged_unlabelled_opere_arte_visiva.csv",
    "../../../dataset/CH_IT/total/all_unlabelled.csv"
]

all_confidences = []

for file_path in input_files:
    df = pd.read_csv(file_path)
    texts = df['descrizione'].astype(str).tolist()
    #results = clf(texts, truncation=True, padding=True, max_length=128)
batch_size = 32
results = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    batch_results = clf(batch, truncation=True, padding=True, max_length=128)
    results.extend(batch_results)
    percent = (i + batch_size) / len(texts) * 100
    print(f"\rProcessing {os.path.basename(file_path)}: {min(percent,100):.1f}% complete", end="")
print()  # for newline after progress
# Get predicted class and confidence
pred_class = [max(r, key=lambda x: x['score'])['label'] for r in results]
confidence = [max(r, key=lambda x: x['score'])['score'] for r in results]
df['predicted_class'] = pred_class
df['confidence'] = confidence
# Save to new file
out_path = file_path.replace("merged_unlabelled_", "labelled_")
df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
avg_conf = df['confidence'].mean()
print(f"Average confidence for {os.path.basename(file_path)}: {avg_conf:.4f}")
all_confidences.extend(df['confidence'].tolist())

overall_avg_conf = sum(all_confidences) / len(all_confidences)
print(f"\nOverall average confidence across all files: {overall_avg_conf:.4f}")