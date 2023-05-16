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

The notebook has example code to load a dataset and apply the tokenizer to pairs of sentences, as per the MRPC dataset/goal of checking if two sentences are paraphrases or not. One highlight is the `token_type_ids` output of the tokenizer, which says which sentence the token belongs to. This is used by the model to distinguish between the two sentences -- but not all models/checkpoints use this, so you need to check the documentation for the model you are using. 

The tokenization can be done in the entire dataset using `tokenizer()`, but this loads the entire dataset into memory. So a much better way is to use the `Dataset.map()`, by defining a funcion (delegate) that tokenizes a single example:

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```

### Dynamic padding

The function that is responsible for putting together samples inside a batch is called a *collate function*. It’s an argument you can pass when you build a PyTorch `DataLoader`, the default being a function that will just convert your samples to PyTorch tensors and concatenate them (recursively if your elements are lists, tuples, or dictionaries). This won’t be possible in our case since the inputs we have won’t all be of the same size. 

We have deliberately postponed the padding, to only apply it as necessary on each batch and **avoid having over-long inputs with a lot of padding**. This will speed up training by quite a bit, but note that if you’re training on a TPU it can cause problems — TPUs prefer fixed shapes, even when that requires extra padding.

To do this in practice, *we have to define a collate function that will apply the correct amount of padding to the items of the dataset we want to batch together*. Fortunately, the Transformers library provides us with such a function via `DataCollatorWithPadding`. It takes a tokenizer when you instantiate it (to know which padding token to use, and whether the model expects padding to be on the left or on the right of the inputs) and will do everything you need:

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

Again see the code in the `process-data.ipynb` notebook.

The same padding can be applied to all sentences by using `return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", truncation=True, max_length=128)`, but this is not recommended as it can lead to a lot of padding, esp in smaller sentences.
