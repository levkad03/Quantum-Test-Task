# Task 1. Natural Language Processing. Named entity recognition

In this task, we need to train a named entity recognition (NER) model for the identification of
mountain names inside the texts.

# Folder structure
- `dataset.ipynb` - notebook with dataset generation
- `train.py` - model training
- `inference.py` - code for model inference

# How to run
- install all dependencies
```
pip install -r requirements.txt
```
- to run model training:
```
python train.py
```
- to try out the model, download the weights from [here](https://drive.google.com/file/d/1D8X0NEcPUf-Qoc-DXs7vhlMK3YfGlTkV/view?usp=sharing), unpack the archive and put "saved_model" folder in folder "task 1"
- to run demo:
```
python inference.py
```
