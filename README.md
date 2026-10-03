# Detecting Noisy Labels Using Checkpoint Ensembles

Code for the paper *Detecting Noisy Labels Using Checkpoint Ensembles*.

## Repository layout

```
├── 00_generate_noise.py                  # synthetic label noise: uniform, flip, instance
├── 01_train_noise_detection.py           # standard training with early stopping; stores the top-k checkpoints
├── 02a_create_filter_logit.py            # ES-Prediction from the stored checkpoints
├── 02b_create_filter_bgss.py             # ES-BGSS from the stored checkpoints
├── 03_train_noise_detection_plateau.py   # standard training with early stopping; predictions are aggregated during training
├── 04_train_clean_model.py               # retraining on the filtered training set
├── 05_evaluate_detection_results.ipynb   # accuracy / F1 / precision / recall of every detector
├── train-utils/                          # utilities package: datasets, models, trainer, callbacks
├── src.py                                # further callbacks
└── run/example.sh                        # example run
```

## Installation

Python 3.10 or newer is required.

```bash
pip install -e train-utils # trainutils + torch, torchvision, numpy, pandas, matplotlib, tqdm
pip install scipy scikit-learn jupyter # used by noise generation and the evaluation notebook
```

## Usage

See `run/example.sh`.
