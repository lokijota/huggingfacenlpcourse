# General notes on chapter 2 - Using Transformers

## Introduction

«The library’s main features are:

- Ease of use: **Downloading, loading, and using a state-of-the-art NLP model for inference can be done in just two lines of code**.

- Flexibility: At their core, **all models are simple PyTorch nn.Module or TensorFlow tf.keras.Model classes** and can be handled like any other models in their respective machine learning (ML) frameworks.

- Simplicity: Hardly any abstractions are made across the library. **The “All in one file” is a core concept: a model’s forward pass is entirely defined in a single file**, so that the code itself is understandable and hackable.

The tokenizer API is the other main component of the `pipeline()` function. **Tokenizers take care of the first and last processing steps, handling the conversion from text to numerical inputs for the neural network, and the conversion back to text when it is needed**. Finally, we’ll show you how to handle sending multiple sentences through a model in a prepared batch, then wrap it all up with a closer look at the high-level `tokenizer()` function.

## Note on Softmax

The softmax function is a mathematical function that converts a vector of real numbers into a vector of probabilities that add up to 1. It is often used in machine learning models, such as neural networks, to perform multi-class classification.

The softmax function works by applying the exponential function to each element of the input vector, and then dividing each element by the sum of all the exponentials. This way, **each element becomes a positive number between 0 and 1, and the whole vector sums to 1**. **The softmax function can be seen as a way of assigning probabilities to different possible outcomes or classes**.

For example, suppose a neural network outputs a vector of three real numbers: (-0.62, 8.12, 2.53). These numbers are not probabilities, and they can be negative or larger than 1. To convert them into probabilities, we can apply the softmax function:

- First, we calculate the exponential (e^x) of each element: (0.54, 3354.73, 12.55).
- Second, we calculate the sum of all the exponentials: 3367.82.
- Third, we divide each element by the sum: (0.00016, 0.9962, 0.0037).

The result is a vector of probabilities: (0.00016, 0.9962, 0.0037). This vector sums to 1, and each element is between 0 and 1. The softmax function has transformed the original vector into a probability distribution that can be interpreted as the confidence of the neural network for each possible class.

## What happens inside the pipeline()

General scheme:

![](pipeline-steps.png)

### 1 - Tokenization

Like other neural networks, Transformer models can’t process raw text directly, so the first step of our pipeline is to convert the text inputs into numbers that the model can make sense of. To do this we use a tokenizer, which will be responsible for:

- Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
- Mapping each token to an integer
- Adding additional inputs that may be useful to the model

All this preprocessing needs to be done **in exactly the same way as when the model was pretrained, so we first need to download that information from the Model Hub**. To do this, we use the `AutoTokenizer` class and its `from_pretrained()` method. Using the checkpoint name of our model, it will automatically fetch the data associated with the model’s tokenizer and cache it.

```
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

Once we have the tokenizer, we can directly pass our sentences to it and we’ll get back a dictionary that’s ready to feed to our model! The only thing left to do is to convert the list of input IDs to tensors.

```
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") # pt = use pytorch tensors in the return
print(inputs)
```

Output looks like this:

```
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

The output itself is a dictionary containing two keys, `input_ids` and `attention_mask`. `input_ids` contains two rows of integers (one for each sentence) that are the unique identifiers of the tokens in each sentence.

### 2 - Apply the model

Download the model's checkpoint (it should be cached already):

```
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

This architecture contains only the base Transformer module: given some inputs, it outputs what we’ll call **hidden states**, also known as **features**. For each model input, we’ll retrieve a *high-dimensional vector representing the contextual understanding of that input by the Transformer model*.

While these hidden states can be useful on their own, *they’re usually inputs to another part of the model, known as the **head***. In Chapter 1, the different tasks could have been performed with the same architecture, but each of these tasks will have a different head associated with it.


The vector output by the Transformer module is usually large. It generally has three dimensions:

- Batch size: The number of sequences processed at a time (2 in our example).
- Sequence length: The length of the numerical representation of the sequence (16 in our example).
- Hidden size: The vector dimension of each model input.

It is said to be “high dimensional” because of the last value (in this case 768).

```
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```
Prints out:

```
torch.Size([2, 16, 768])
```

Outputs of Transformers-library models behave like namedtuples/dictionaries. You can access the elements by attributes (like we did) or by key (`outputs["last_hidden_state"]`), or even by index if you know exactly where the thing you are looking for is (`outputs[0]`).

#### Model heads - Making sense out of numbers

*The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension*. They are usually composed of one or a few linear layers:

![](model-heads.png)

The output of the Transformer model is sent directly to the model head to be processed.

In this diagram, the model is represented by its embeddings layer and the subsequent layers. **The embeddings layer converts each input ID in the tokenized input into a vector that represents the associated token. The subsequent layers manipulate those vectors using the attention mechanism to produce the final representation of the sentences.**

There are many different architectures available in the Transformers library, with each one designed around tackling a specific task. Here is a non-exhaustive list:

- *Model (retrieve the hidden states)
- *ForCausalLM
- *ForMaskedLM
- *ForMultipleChoice
- *ForQuestionAnswering
- *ForSequenceClassification
- *ForTokenClassification

For our example, we will need a model with a sequence classification head (to be able to classify the sentences as positive or negative). So, we won’t actually use the `AutoModel` class, but `AutoModelForSequenceClassification`:

```
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs) # **inputs unpacks the dictionary, and passes the values of each dictionary as a named argument to the model

print(outputs.logits.shape)
```

This prints out: `torch.Size([2, 2])`

### 3 - Postprocessing the output

The values we get as output from our model don’t necessarily make sense by themselves.

```
print(outputs.logits)
```

```
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```

Each row on the tensor above represents a sentence, each column represents a possible label (positive or negative). The values in the tensor are called logits, and they represent the model’s predictions for each label. The higher the value, the more likely the model thinks the input sentence is of that label. Remember: Those are not probabilities but logits, the raw, unnormalized scores outputted by the last layer of the model. To be converted to probabilities, they need to go through a SoftMax layer (**all Transformers-library models output the logits**, as the loss function for training will generally fuse the last activation function, such as SoftMax, with the actual loss function, such as cross entropy):

```
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

```
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```

So the model predicted 0.04 for the first sentence being negative and 0.96 for it being positive. For the second sentence, it predicted 0.99 for it being negative and 0.0005 for it being positive. To get the labels corresponding to each position, we can inspect the id2label attribute of the model config:

```	
model.config.id2label
```

```	
{0: 'NEGATIVE', 1: 'POSITIVE'}
```	

So we have successfully reproduced the three steps of the pipeline:
- preprocessing with tokenizers
- passing the inputs through the model
-  postprocessing

The following sections go deeper into these topics.

## Models

- The `AutoModel` class and all of its relatives are actually simple wrappers over the wide variety of models available in the library. It’s a **clever wrapper as it can automatically guess the appropriate model architecture for your checkpoint, and then instantiates a model with this architecture**. However, if you know the type of model you want to use, you can use the class that defines its architecture directly.

- **Note** - Check `model.ipynb` for code

For example:

```	
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

# Model is randomly initialized!
```

The model can be used in this state, but it will output gibberish; it needs to be trained first. 

Loading a Transformer model that is already trained is simple:

```
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

As you saw earlier, we could replace `BertModel` with the equivalent AutoModel class. We’ll do this from now on as this produces checkpoint-agnostic code; if your code works for one checkpoint, it should work seamlessly with another. This applies even if the architecture is different, as long as the checkpoint was trained for a similar task (for example, a sentiment analysis task).

In the code sample above we didn’t use `BertConfig`, and instead loaded a pretrained model via the `bert-base-cased` identifier. This is a model checkpoint that was trained by the authors of BERT themselves.

This model is now initialized with all the weights of the checkpoint. It can be used directly for inference on the tasks it was trained on, and it can also be fine-tuned on a new task. By training with pretrained weights rather than from scratch, we can quickly achieve good results.

The weights have been downloaded and cached (so future calls to the `from_pretrained`() method won’t re-download them) in the cache folder. On windows, on my case, that's `C:\Users\joamart\.cache\huggingface\hub`.

When you save a model (`model.save_pretrained("directory_on_my_computer")`), you get a json file and a bin file. The pytorch_model.bin file is known as the **state dictionary**; it contains all your model’s weights. The two files go hand in hand; the configuration is necessary to know your model’s architecture, while the model weights are your model’s parameters.

### Using a Transformer model for inference

Now that you know how to load and save a model, let’s try using it to make some predictions. **Transformer models can only process numbers — numbers that the tokenizer generates**. But before we discuss tokenizers, let’s explore what inputs the model accepts: **tensors**.

Tokenizers can take care of casting the inputs to the appropriate framework’s tensors, but to help you understand what’s going on, we’ll take a quick look at what must be done before sending the inputs to the model.

Let’s say we have a couple of sequences:

```
sequences = ["Hello!", "Cool.", "Nice!"]
```

The tokenizer converts these to vocabulary indices which are typically called input IDs. Each sequence is now a list of numbers! The resulting output is:

```
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
```

This is a list of encoded sequences: a list of lists. Tensors only accept matrices, so it has to be converted to a tensor:

```
import torch

model_inputs = torch.tensor(encoded_sequences)
```

While the model accepts a lot of different arguments, only the input IDs are necessary.

## Tokenizers

Tokenizers transform input text into "numbers" that can be processed by a model. There are three types:

- Word based
- Character based
- Subword based

There are a lot of ways of doing this. The goal is to find the most *meaningful representation* — that is, the one that makes the most sense to the model — and, if possible, the smallest representation.

### Word-based tokenizers

- split based on spaces or punctuation
- each word has a specific id assigned to it
- limits: similar words, like dog and dogs, are similar but will have totally different Ids. So the model won't know the relationship between them.
- the ids are usually assigned based on the frequency of the word in the training corpus. Also, the vocabulary size can be huge, which can be a problem for the model.
- we can have the larger words be ignored, eg consider only the most used 10k words. Any new word will be assigned the same id as the last word in the vocabulary. This is the "out of vocabulary" token, also represented as `[UNK]`, and this results in loss of meaningful information.

One way to reduce the amount of unknown tokens is to go one level deeper, using a character-based tokenizer.

### Character-based tokenizers

- assign numbers to characters, also considering accented words and special characters (~256 for western languages).
- this results in much fewer "out of vocabulary" signs, and is also able to represent mis-spelled words
- but character tokens hold much less information than words tokens
- also, a lot more tokens will be needed to represent the same text, which can be a problem for the model. this will also reduce the size of the context/size of input to the model
- we’ll end up with a very large amount of tokens to be processed by our model: whereas a word would only be a single token with a word-based tokenizer, it can easily turn into 10 or more tokens when converted into characters.
- this type of tokenization has however been succesful in the past (for some types of problems) 

To get the best of both worlds, we can use a third technique that combines the two approaches: subword tokenization.

### Subword-based tokenizers

Middle ground between the two previous approaches. The idea is to split the words into smaller parts, and then assign ids to these parts. This way, we can represent the words that are not in the vocabulary as a combination of subwords that are in the vocabulary.

Principles:
- Frequently used words should not be split into subwords
- Rare words should be decomposed into meaningful subwords

Eg, "dog" should be a single token, but "dogs" should be "dog" + "s", so that we keep the meeaning of "dog". Other cases are having tokens for root words -- eg, tokenization = "token" (root) + "ization" (suffix, completion of the word).

tokenization = token + ##ization. ## means "part of a word" for BERT's tokenizer. For Alberta the _ is used instead (and most word-based tokenizers)

There are several tokenizers in use today, eg:
- WordPiece (Bert, DistilBert)
- Unigram (XLNet, Albert)
- Byte-Pair Encoding (GPT-2, RoBERTa)

Most models with SOTA result use subword tokenization.

Go check code in `tokenizers.ipynb`.

Steps inside tokenizer are *roughly*:

1. Raw text
2. Convert to tokens (break apart words)
3. Add special tokens (eg, CLS, SEP, ##...)
4. Convert to ids

When you create a tokenizer, you download the vocabulary file, which corresponds to the mapping between tokens and ids for the model you want to use.
This happens when we instantiate it with `AutoTokenizer/BertTokenizer/....from_pretrained()`.

At the end, the `tokenizer.decode()` method is used to convert the ids back to text. This is also used to convert the output of a model into readable text.

## Handling multiple sequences

This is more of a "nice to know" chapter, but it's important to understand how the model handles multiple sequences. It talks about the attention mask in the inputs to tranformers models, and how to handle individual inputs instead of "batches" of 2+ sentences.

Text is here: https://huggingface.co/learn/nlp-course/chapter2/5?fw=pt and code is in `multiple_sequences.ipynb`.

### Sending a single sentence vs a batch

Remember to add in additional list nesting.

Batching is the act of sending multiple sentences through the model, all at once. If you only have one sentence, you can just build a batch with a single sequence:

```
batched_ids = [ids]
```

If you don't do it, you'll have an error like the one shown in the notebook.

### Padding when using a batch with sentences of different dimensions

When we have sentences in a batch of different dimension, the rows of Id's will have different sizes:

```
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
```

This is a problem for the model, because it expects a tensor with a fixed size. To solve this, we add padding to the sequences with less tokens, so that they have the same size as the largest sequence in the batch.

```
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```

The padding token ID can be found in `tokenizer.pad_token_id`.

### Attention masks

There is another important note. If you pass in just the padded contents, the model will still pay attention to the padding tokens **and use them as inputs to the model**, which is not what we want. To avoid this, we use an attention mask, which is a tensor of 0s and 1s, where 0s correspond to the padding tokens:

```
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

This is because the key feature of Transformer models is **attention layers that contextualize each token**. These will take into account the padding tokens since they attend to all of the tokens of a sequence. To get the same result when passing individual sentences of different lengths through the model or when passing a batch with the same sentences and padding applied, **we need to tell those attention layers to ignore the padding tokens. This is done by using an attention mask.**

**Attention masks** are tensors *with the exact same shape as the input IDs tensor*, *filled with 0s and 1s*: 1s indicate the corresponding tokens should be attended to, and 0s indicate the corresponding tokens should not be attended to (i.e., they should be ignored by the attention layers of the model).

## Putting it all together

The `tokenizer()` function can handle the tokenization for us with convenience, while providing the required degree of finer control.

```
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

The `tokenizer()` function handles:

- single sentences
- lists of sentences
- pad in different ways:
    ```
    # Will pad the sequences up to the maximum sequence length
    model_inputs = tokenizer(sequences, padding="longest")

    # Will pad the sequences up to the model max length
    # (512 for BERT or DistilBERT)
    model_inputs = tokenizer(sequences, padding="max_length")

    # Will pad the sequences up to the specified max length
    model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
    ```
- can also do truncation:
    ```
    # Will truncate the sequences that are longer than the model max length (512 for BERT or DistilBERT)
    model_inputs = tokenizer(sequences, truncation=True)

    # Will truncate the sequences that are longer than the specified max length
    model_inputs = tokenizer(sequences, max_length=8, truncation=True)
    ```
- Can handle the conversion to specific framework tensors, which can then be directly sent to the model. - "pt" returns PyTorch tensors, "tf" returns TensorFlow tensors, and "np" returns NumPy arrays:
    ```
    # Returns PyTorch tensors
    model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

    # Returns TensorFlow tensors
    model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

    # Returns NumPy arrays
    model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
    ```
- Handles special tokens used by the different models as separators
    ```
    sequence = "I've been waiting for a HuggingFace course my whole life."

    model_inputs = tokenizer(sequence)
    print(model_inputs["input_ids"])

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)

    print(tokenizer.decode(model_inputs["input_ids"]))
    print(tokenizer.decode(ids))
    ```
    Print out (note "CLS" and "SEP" -- which were used to train this **specific** model):
    ```
    [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
    [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]

    "[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
    "i've been waiting for a huggingface course my whole life."
    ```

So here's a final complete example:

```
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
print(output.logits)
```

