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

# This is new -- indicating what the labels are ("Positive") for the 2 new sentences
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

The function that is responsible for putting together samples inside a batch is called a *collate function*. It’s an *argument* you can pass when you build a PyTorch `DataLoader`, the default being a function that will just convert your samples to PyTorch tensors and concatenate them (recursively if your elements are lists, tuples, or dictionaries). This won’t be possible in our case since the inputs we have won’t all be of the same size.

*Note:* In PyTorch, DataLoader is a built-in class that provides an efficient and flexible way to load data into a model for training or inference. It is particularly useful for handling large datasets that cannot fit into memory, as well as for performing data augmentation and preprocessing.

We have deliberately postponed the padding, to only apply it as necessary on each batch and **avoid having over-long inputs with a lot of padding**. This will speed up training by quite a bit, but note that if you’re training on a TPU it can cause problems — TPUs prefer fixed shapes, even when that requires extra padding.

To do this in practice, *we have to define a collate function that will apply the correct amount of padding to the items of the dataset we want to batch together*. Fortunately, the Transformers library provides us with such a function via `DataCollatorWithPadding`. It takes a tokenizer when you instantiate it (to know which padding token to use, and whether the model expects padding to be on the left or on the right of the inputs) and will do everything you need:

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

Again see the code in the `process-data.ipynb` notebook.

The same padding can be applied to all sentences by using `return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", truncation=True, max_length=128)`, but this is not recommended as it can lead to a lot of padding, esp. in smaller sentences.

## Fine-tuning a model with the Trainer API

Transformers provides a Trainer class to help you fine-tune any of the pretrained models it provides on your dataset. Once you’ve done all the data preprocessing work in the last section, you have just a few steps left to define the Trainer. The hardest part is likely to be preparing the environment to run `Trainer.train()`, as it will run very slowly on a CPU. 

See `fine-tuning.ipynb` for the code.

### Training

Steps:

1. The first step before we can define our Trainer is to define a `TrainingArguments` class that will contain all the hyperparameters the Trainer will use for training and evaluation. The only argument you have to provide is a directory where the trained model will be saved, as well as the checkpoints along the way. For all the rest, you can leave the defaults, which should work pretty well for a basic fine-tuning.

2. The second step is to define our model (which will be FT'd on a specific task)

3. The third step is to define our Trainer. We need to pass it the model, the training arguments, and the training and validation datasets. We also need to pass it our data collator, which will be used to dynamically pad the inputs of our dataset to the maximum length in the batch.

4. The last step is to start the training! We just need to call `Trainer.train()` and wait for the results.

The last step above will start the fine-tuning and report the training loss every 500 steps. It won’t, however, tell you how well your model is performing. This is because:

- We didn’t tell the Trainer to evaluate during training by setting `evaluation_strategy` to either "`steps`" (evaluate every `eval_steps`) or "`epoch`" (evaluate at the end of each epoch).
- We didn’t provide the Trainer with a `compute_metrics()` function to calculate a metric during said evaluation (otherwise the evaluation would just have printed the loss, which is not a very intuitive number).


**Important**: I had to do `pip install accelerate`, ran into some issues, as per https://github.com/huggingface/transformers/issues/22816 . Solution was to uninstall transformes, then reinstall and also added pip install accelerate. The FT took about 5 minutes on my laptop.

### Evaluation

How can we build a useful `compute_metrics()` function and use it the next time we train? The function must take an `EvalPrediction` object (which is a named tuple with a predictions field and a `label_ids` field) and will return a dictionary mapping strings to floats (the strings being the names of the metrics returned, and the floats their values). To get some predictions from our model, we can use the `Trainer.predict()` command:

```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

```
(408, 2) (408,)
```

The output of the `predict()` method is another named tuple with three fields: `predictions`, `label_ids`, and `metrics`. The `metrics` field will just contain the loss on the dataset passed, as well as some time metrics (how long it took to predict, in total and on average). Once we complete our `compute_metrics()` function and pass it to the Trainer, that field will also contain the metrics returned by `compute_metrics()`.

**Note**: needed to install sklearn to get the metrics. `pip install scikit-learn` (not `sklearn` as an error message says)

