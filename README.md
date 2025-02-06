# DL-GPT: Building a Large Language Model from Scratch

## Overview
DL-GPT is a deep learning project focused on understanding and replicating the architecture of GPT-2 (124M parameters) from scratch. The project follows the principles outlined in *Build a Large Language Model (From Scratch)* by Sebastian Raschka, covering tokenization, transformer-based architectures, training methodologies, and model evaluation.

## Project Objectives
- Gain a deep understanding of large language models (LLMs)
- Implement key components such as tokenization, embeddings, self-attention, and transformer blocks
- Train and fine-tune a GPT model for classification and assistant tasks


## Training & Fine-Tuning
- **Pre-training**: Small dataset (*The Verdict* by Edith Wharton)
- **Fine-Tuning**:
  - Spam classification using `sms spam collection` dataset
  - Assistant model trained on Raschka's instruction dataset

## Deployment
A simple web UI was built for model interaction:
- **Frontend**: Vue.js 3
- **Backend**: Flask API
- **Containerization**: Docker (Future work)

## Results & Learnings
- Successfully implemented a GPT-2 model with pre-training and fine-tuning.
- Achieved 96.15% accuracy on spam classification.
- Developed an assistant model demonstrating instruction-following capabilities.
- Gained hands-on experience with PyTorch and transformer models.

## References
- *Build a Large Language Model (From Scratch)* - Sebastian Raschka
- [GPT-2 by OpenAI](https://github.com/openai/gpt-2)
- [LLMs from Scratch Repository](https://github.com/rasbt/LLMs-from-scratch)

