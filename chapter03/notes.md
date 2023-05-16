# Fine-tuning a Pre-Trained Model

## Processing the data

Check code in `processing_data.ipynb`.

Here is how we would train a sequence classifier on one batch in PyTorch:

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

### Loading a dataset from the Hub

The Hub doesn’t just contain models; it also has multiple datasets in lots of different languages. You can browse the datasets here - https://huggingface.co/datasets, and we recommend you try to load and process a new dataset once you have gone through this section. But for now, let’s focus on the MRPC dataset! This is one of the 10 datasets composing the GLUE benchmark, which is an academic benchmark that is used to measure the performance of ML models across 10 different text classification tasks.

 The MRPC (Microsoft Research Paraphrase Corpus) dataset consists of 5801 pairs of sentences, with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing): https://huggingface.co/datasets/glue

 How to load a dataset:

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc") # "glue" is the name of the dataset, mrpc"
# https://huggingface.co/docs/datasets/v2.12.0/en/package_reference/loading_methods#datasets.load_dataset
raw_datasets
```

```
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

We get a DatasetDict object which contains the training set, the validation set, and the test set. Each of those contains several columns (sentence1, sentence2, label, and idx) and a variable number of rows, which are the number of elements in each set.

This command downloads and caches the dataset, by default in `C:\Users\joamart\.cache\huggingface\datasets` on Windows.

