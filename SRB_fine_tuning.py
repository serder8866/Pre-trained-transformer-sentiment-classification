# %%
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# %%
file_path = 'SerbMR-2C.csv'
serbmr_data = pd.read_csv(file_path, encoding='utf-8')
# Columns renamed for consistency
serbmr_data = serbmr_data.rename(columns={"Text": "text", "class-att": "label"})

# Map labels to integers
label_mapping = {'POSITIVE': 1, 'NEGATIVE': 0}
serbmr_data['label'] = serbmr_data['label'].map(label_mapping)

train_data, val_data = train_test_split(serbmr_data, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)


# %%
# BERTić training
model_name_bertić = "classla/bcms-bertic"
tokenizer_bertić = AutoTokenizer.from_pretrained(model_name_bertić)
model_bertić = AutoModelForSequenceClassification.from_pretrained(model_name_bertić, num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer_bertić(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args_bertić = TrainingArguments(
    output_dir="./results_bertić_SRB",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True
)

trainer_bertić = Trainer(
    model=model_bertić,
    args=training_args_bertić,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer_bertić.train()

trainer_bertić.model.save_pretrained("saved_models/bertic_model_SRB")
tokenizer_bertić.save_pretrained("saved_models/bertic_tokenizer_SRB")

# %%
# XLM-R training
model_name_xlmr = "xlm-roberta-base"
tokenizer_xlmr = AutoTokenizer.from_pretrained(model_name_xlmr)
model_xlmr = AutoModelForSequenceClassification.from_pretrained(model_name_xlmr, num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer_xlmr(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

train_dataset_xlmr = train_dataset.map(tokenize_function, batched=True)
val_dataset_xlmr = val_dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args_xlmr = TrainingArguments(
    output_dir="./results_xlmr_new",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True
)

trainer_xlmr = Trainer(
    model=model_xlmr,
    args=training_args_xlmr,
    train_dataset=train_dataset_xlmr,
    eval_dataset=val_dataset_xlmr,
)

trainer_xlmr.train()

model_xlmr.save_pretrained("saved_models/xlmr_model_srb")
tokenizer_xlmr.save_pretrained("saved_models/xlmr_tokenizer_srb")



