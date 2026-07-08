# Parler-TTS Fine-tuning Notebook

This repository contains a Jupyter notebook for fine-tuning the Parler-TTS model on a single speaker dataset.

## Overview

This notebook demonstrates how to fine-tune the Parler-TTS Mini v0.1 model on a single speaker dataset using the Data-Speech library. It includes:

1. Dataset preparation and annotation using Data-Speech
2. Fine-tuning the Parler-TTS model
3. Inference with the fine-tuned model

## File Structure

- `Finetuning_Parler_TTS_on_a_single_speaker_dataset.ipynb`: The main Jupyter notebook containing the complete fine-tuning workflow

## Getting Started

### Prerequisites

- Python 3.8 or higher
- GPU with at least 16GB VRAM
- Hugging Face account with write access

### Installation

1. Clone the Parler-TTS repository:
```bash
git clone https://github.com/huggingface/parler-tts.git
cd parler-tts
```

2. Install required packages:
```bash
pip install -e .[train]
pip install jupyter
```

### Running the Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `Finetuning_Parler_TTS_on_a_single_speaker_dataset.ipynb`

3. Follow the step-by-step instructions in the notebook

## Usage

This notebook is designed to be run in a Jupyter environment. It demonstrates fine-tuning on the Jenny TTS dataset (Irish female speaker) but can be adapted for other single-speaker datasets.

## Adaptation for Your Own Dataset

To use this notebook with your own dataset:
1. Prepare your dataset in the same format as the Jenny TTS dataset
2. Upload your dataset to Hugging Face Hub
3. Modify the dataset names in the notebook
4. Adjust training parameters as needed

## Resources

- [Parler-TTS Documentation](https://huggingface.co/docs/transformers/model_doc/parler-tts)
- [Data-Speech Library](https://github.com/huggingface/dataspeech)