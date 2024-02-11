## LoRD: *Lo*w-*R*ank *D*ecomposition of finetuned Large Language Models

This repository contains code for extracting LoRA adapters from finetuned `transformers` models, using Singular Value Decomposition (SVD). 

LoRA (*Lo*w-*R*ank *A*daptation) is a technique for parameter-efficient fine-tuning of large language models. The technique presented here allows extracting [PEFT](https://huggingface.co/docs/peft/index) compatible Low-Rank adapters from full fine-tunes or merged model.

# Getting started

Everything you need to extract and publish your LoRA adapter is available in the [`LoRD.ipynb`](LoRD.ipynb) notebook.

Running the notebook on [Colab](https://colab.research.google.com/github/thomasgauthier/LoRD/blob/main/LoRD.ipynb) is the easiest way to get started.

# Special thanks

Thanks to @kohya_ss for their [prior work](https://github.com/bmaltais/kohya_ss/blob/master/networks/extract_lora_from_models.py) on LoRA extraction for Stable Diffusion.