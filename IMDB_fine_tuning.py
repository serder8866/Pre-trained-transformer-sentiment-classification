# %%
# Training the models on the translated IMDB dataset

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# %%
imdb_data = pd.read_csv('Properly_Romanized_IMDB.csv', encoding='utf-8')

# Rename columns to fit Hugging Face format
imdb_data = imdb_data.rename(columns={"review": "text", "sentiment": "label"})

# Convert labels to integers
label_mapping = {'positive': 1, 'negative': 0}
imdb_data['label'] = imdb_data['label'].map(label_mapping)

train_data, val_data = train_test_split(imdb_data, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)


# %%


# Training BERTić
model_name = "classla/bcms-bertic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_bertic_IMDB",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True  # This was necessary due to hardware limitations
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained("saved_models/bertic_model_IMDB")
tokenizer.save_pretrained("saved_models/bertic_tokenizer_IMDB")


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

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Same training arguments as for BERTic
training_args_xlmr = TrainingArguments(
    output_dir="./results_xlmr_IMDB",
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
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer_xlmr.train()

model_xlmr.save_pretrained("saved_models/xlmr_model_IMDB")
tokenizer_xlmr.save_pretrained("saved_models/xlmr_tokenizer_IMDB")

# %%



