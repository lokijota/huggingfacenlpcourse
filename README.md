# Hugging Face NLP Course

***Note**: these are my personal notes, which I took while doing the HF NLP course. They are NOT a replacement or rewrite of the course, they are just notes supporting my learning.* 

- Course: https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt
- Notebooks: https://github.com/huggingface/notebooks
- Transformers Library documentation: https://huggingface.co/docs/transformers/index
- Gradio (owned by HF): https://github.com/gradio-app/gradio
- Codecarbon for CO2 emissions tracking - https://github.com/mlco2/codecarbon

## Environment Setup

### Install CUDA

- Start with CUDA (RTX A2000 is supported - https://developer.nvidia.com/cuda-gpus) - https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network
- Installed v12.1

### Install PyTorch

- https://pytorch.org/get-started/locally/#start-locally gives you the command to run: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
- Picked CUDA 11.8, which is supposed to work with CUDA 12.1 (as per https://discuss.pytorch.org/t/install-pytorch-with-cuda-12-1/174294/3)

### Install base python libraries

```
conda create --name hfnlp python=3.9
conda activate hfnlp
pip install numpy
pip install ipykernel
conda install -c conda-forge ipywidgets

# To avoid this error https://github.com/huggingface/transformers/issues/21858
pip install chardet
pip install cchardet

# as per the first sample notebook
pip install datasets evaluate transformers[sentencepiece] 
```

## Notes per chapter

- [Chapter 1 - Transformer Models](./chapter01/notes.md)
- [Chapter 2 - Using Transformers](./chapter02/notes.md)
- [Chapter 3 - Fine-Tuning a Pretrained model](./chapter03/notes.md)

- [Spin-off topics/readings](./SpinOffs/README.md)

## Other tutorials to do

- Fine-tuning with custom datasets: https://huggingface.co/transformers/v3.2.0/custom_datasets.html - this uses an older version of transformers (v3.2.0) but it does fine-tuning with datasets not stored in HF, which I want to check out.

- FAISS/HNSW - To check too: https://www.pinecone.io/learn/hnsw/

- Langchain tutorial (shared by NS) - https://www.youtube.com/watch?v=aywZrzNaKjs

- Diffusers HF library - https://huggingface.co/docs/diffusers/index

## Bibtext reference to HF course:

```
@misc{huggingfacecourse,
  author = {Hugging Face},
  title = {The Hugging Face Course, 2022},
  howpublished = "\url{https://huggingface.co/course}",
  year = {2022},
  note = "[Online; accessed 2023/04/25]"
}
```
