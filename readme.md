# ALRMR-GEC: Adjusting Learning Rate Based on Memory Rate to Optimize the Edit Scorer for Grammatical Error Correction
Code for AAAI2025 paper **ALRMR-GEC: Adjusting Learning Rate Based on Memory Rate to Optimize the Edit Scorer for Grammatical Error Correction** that provides a
novel approach to adjust learning rate on grammatical error correction.

## Installation
* `conda create -n ALRMR-GEC python=3.8`
* `conda activate ALRMR-GEC`
* `pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121`
* `pip install -r requirements.txt`

Our work is based on https://github.com/AlexeySorokin/EditScorer.git. The core of our work is to introduce a memory rate to automatically adjust the learning rate. Therefore, the experimental setup, model acquisition, dataset acquisition, and model training commands can follow the original work.


