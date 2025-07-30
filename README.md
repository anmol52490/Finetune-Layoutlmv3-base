
# Fine-tuning LayoutLMv3-base for Hierarchical Heading Detection

This repository contains the codebase for fine-tuning the [LayoutLMv3-base](https://huggingface.co/microsoft/layoutlmv3-base) model on a modified version of the [DocLayNet dataset]([https://github.com/vis-nlp/DocLayNet](https://huggingface.co/datasets/pierreguillou/DocLayNet-base)). The goal is to detect **hierarchical headings (H1, H2, H3, etc.)** from document PDFs using layout-aware multimodal learning.

## 📌 Project Overview

The project includes:
- Data cleaning scripts for filtering corrupted image-annotation pairs.
- Heuristic preprocessing to label heading hierarchy.
- A training pipeline to fine-tune the LayoutLMv3 model using HuggingFace `Trainer`.
- Model saving and evaluation logic.

## ⚙️ Repository Structure

```

├── cleanup.py         # Removes corrupted JSON/image files based on logs
├── create\_dataset.py  # Validates and converts raw DocLayNet into HuggingFace format
├── labels.py          # Alternate dataset creation script for quick testing
├── finetune.py        # Fine-tuning script for LayoutLMv3-base on the processed dataset

```

## 📦 Model and Dataset Access

> **Note:** This repo contains only the **codebase**.  
> The **preprocessed dataset** (based on DocLayNet) used for training is **not included** due to its large size.

If you want access to the dataset or to reproduce the training pipeline:
- Please contact **[@anmol52490](https://github.com/anmol52490)**.


## 🛠 Tools & Libraries

- Python, PyTorch, HuggingFace Transformers, Datasets
- LayoutLMv3-base
- PIL, TQDM, concurrent.futures

## 📩 Contact

If you have any questions or need access to the processed dataset:
📬 Reach out to: **[anmol52490](https://github.com/anmol52490)**
