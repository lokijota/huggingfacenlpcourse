# Hugging Face NLP Course

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

# as per the first sample notebook
pip install datasets evaluate transformers[sentencepiece] 
```

## Notes per chapter

- [Chapter1](chapter01\notes.md)

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
