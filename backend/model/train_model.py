import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import joblib

# Load dataset
df = pd.read_csv("ai-mood-music-recommender\data\emotions.csv")

# Drop any NaNs just in case
df = df.dropna(subset=['text', 'label'])

# Encode labels
df['label'] = df['label'].astype('category')
df['labels'] = df['label'].cat.codes  # 💡 Hugging Face Trainer expects 'labels' as column name
label2id = dict(enumerate(df['label'].cat.categories))
id2label = {v: k for k, v in label2id.items()}
joblib.dump((label2id, id2label), 'label_mappings.pkl')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create HF Dataset
hf_dataset = Dataset.from_pandas(df[['text', 'labels']])  # Only keep what we need
hf_dataset = hf_dataset.train_test_split(test_size=0.1)

# Tokenization
def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

hf_dataset = hf_dataset.map(tokenize_fn, batched=True)

# Set format for PyTorch
hf_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label2id))

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=50,
    logging_dir="./logs",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset["train"],
    eval_dataset=hf_dataset["test"],
    tokenizer=tokenizer
)

# Train!
trainer.train()

# Save
model.save_pretrained("model")
tokenizer.save_pretrained("model")
