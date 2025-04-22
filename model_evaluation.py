# %%
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import matplotlib.pyplot as plt

# The testing is done only on Serbian data
file_path = 'SerbMR-2C.csv'
serbmr_data = pd.read_csv(file_path, encoding='utf-8')

# %%
# Rename columns for consistency
serbmr_data = serbmr_data.rename(columns={"Text": "text", "class-att": "label"})

# Map labels to integers
label_mapping = {'POSITIVE': 1, 'NEGATIVE': 0}
serbmr_data['label'] = serbmr_data['label'].map(label_mapping)

train_data, val_data = train_test_split(serbmr_data, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# %%


# Load the models
model_xlmr_srb = AutoModelForSequenceClassification.from_pretrained("saved_models/xlmr_model_srb")
tokenizer_xlmr_srb = AutoTokenizer.from_pretrained("saved_models/xlmr_tokenizer_srb")

model_xlmr_IMDB = AutoModelForSequenceClassification.from_pretrained("saved_models/xlmr_model_IMDB")
tokenizer_xlmr_IMDB = AutoTokenizer.from_pretrained("saved_models/xlmr_tokenizer_IMDB")

model_bertić_IMDB = AutoModelForSequenceClassification.from_pretrained("saved_models/bertic_model_IMDB")
tokenizer_bertić_IMDB = AutoTokenizer.from_pretrained("saved_models/bertic_tokenizer_IMDB")

model_bertić_srb = AutoModelForSequenceClassification.from_pretrained("saved_models/bertic_model_srb")
tokenizer_bertić_srb = AutoTokenizer.from_pretrained("saved_models/bertic_tokenizer_srb")


# %%

# Evaluate function for accuracy, precision, recall and F1
def evaluate_model(model, tokenizer, dataset):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    true_labels = []
    
    for example in dataset:
        inputs = tokenizer(
            example['text'], 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        
        predictions.append(predicted_label)
        true_labels.append(example['label'])

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    return accuracy, precision, recall, f1


# %%

# Evaluate the models
acc_xlmr_srb, prec_xlmr_srb, rec_xlmr_srb, f1_xlmr_srb = evaluate_model(model_xlmr_srb, tokenizer_xlmr_srb, val_dataset)
acc_xlmr_IMDB, prec_xlmr_IMDB, rec_xlmr_IMDB, f1_xlmr_IMDB = evaluate_model(model_xlmr_IMDB, tokenizer_xlmr_IMDB, val_dataset)

acc_bertić_srb, prec_bertić_srb, rec_bertić_srb, f1_bertić_srb = evaluate_model(model_bertić_srb, tokenizer_bertić_srb, val_dataset)
acc_bertić_IMDB, prec_bertić_IMDB, rec_bertić_IMDB, f1_bertić_IMDB = evaluate_model(model_bertić_IMDB, tokenizer_bertić_IMDB, val_dataset)


# %%


results = pd.DataFrame({
    "Model": ["XLM-R SRB", "XLM-R IMDB", "BERTić SRB", "BERTić IMDB"],
    "Accuracy": [acc_xlmr_srb, acc_xlmr_IMDB, acc_bertić_srb, acc_bertić_IMDB],
    "Precision": [prec_xlmr_srb, prec_xlmr_IMDB, prec_bertić_srb, prec_bertić_IMDB],
    "Recall": [rec_xlmr_srb, rec_xlmr_IMDB, rec_bertić_srb, rec_bertić_IMDB],
    "F1 Score": [f1_xlmr_srb, f1_xlmr_IMDB, f1_bertić_srb, f1_bertić_IMDB]
})

print(results)
# %%

# Create a confusion matrix
def plot_confusion_matrix(model, tokenizer, dataset, model_name="Model"):
    predictions = []
    true_labels = []

    for example in dataset:
        inputs = tokenizer(
            example['text'], 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        
        predictions.append(predicted_label)
        true_labels.append(example['label'])
    
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NEGATIVE", "POSITIVE"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


# %%

# Create a confusion matrix for each model
plot_confusion_matrix(model_xlmr_srb, tokenizer_xlmr_srb, val_dataset, model_name="XLM-R SRB")
plot_confusion_matrix(model_xlmr_IMDB, tokenizer_xlmr_IMDB, val_dataset, model_name="XLM-R IMDB")
plot_confusion_matrix(model_bertić_srb, tokenizer_bertić_srb, val_dataset, model_name="BERTić SRB")
plot_confusion_matrix(model_bertić_IMDB, tokenizer_bertić_IMDB, val_dataset, model_name="BERTić IMDB")