
# Pre-trained Transformer Sentiment Classification

This repository provides a framework for performing sentiment analysis using pre-trained transformer models. It includes scripts for data preprocessing, model fine-tuning, and evaluation.

## Overview

The project focuses on:

- **Preprocessing**: Cleaning and preparing English and Serbian datasets for training.
- **Fine-tuning**: Adapting pre-trained transformer models to the sentiment classification task.
- **Evaluation**: Assessing model performance using standard classification metrics.

## Repository Structure

- `eng_preprocessing.py`: Preprocesses the English IMDB dataset.
- `IMDB_fine_tuning.py`: Fine-tunes a transformer model on the English dataset.
- `SRB_fine_tuning.py`: Fine-tunes a transformer model on the Serbian dataset.
- `model_evaluation.py`: Evaluates the performance of the fine-tuned models.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/serder8866/Pre-trained-transformer-sentiment-classification.git
   cd Pre-trained-transformer-sentiment-classification
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Preprocessing

- **English Dataset**:

  ```bash
  python eng_preprocessing.py
  ```

- **Serbian Dataset**:

  *(Assuming a Serbian preprocessing script is available or included in `SRB_fine_tuning.py`.)*

### Fine-tuning

- **English Model**:

  ```bash
  python IMDB_fine_tuning.py
  ```

- **Serbian Model**:

  ```bash
  python SRB_fine_tuning.py
  ```

### Evaluation

After fine-tuning, evaluate the models:

```bash
python model_evaluation.py
```

Make sure model checkpoints and dataset paths are properly set in the scripts.

## Datasets

- **English**: [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Serbian**: [SerbMR-2C Dataset](https://github.com/serder8866/SerbMR-2C)

Ensure datasets are downloaded and placed in the correct directories expected by the scripts.

## Models

This project uses pre-trained transformer models from Hugging Face’s Transformers library. Models such as `bert-base-uncased`, `xlm-roberta-base`, or domain-specific models like `BERTić` can be specified.

## Evaluation Metrics

- **Accuracy**: Overall correctness.
- **Precision**: Correct positive predictions over total positive predictions.
- **Recall**: Correct positive predictions over all actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [SerbMR-2C Dataset](https://github.com/serder8866/SerbMR-2C)
