# AAML_S25

## General information

This project was done for the course AAML at LMU Munich by:

* Benedikt Pilger
* Ilir Hajrullahu

## Setting up the environment
Note: This was only tested on 2 Windows PCs (Windows 10 and 11).
In order to directly run the scripts you have to set up a conda environment with Python 3.10.
Run this command in the environment:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
After that, in the highest level of the repository you will find a requirements.txt.
Please install the requirements in the environment.
```bash
pip install -r requirements.txt
```

## Project structure

There are 3 main folders:
* pacman
* snake
* pacman_and_snake

Each of these folders is divided into deep-q-learning and ppo. Inside these folders you will find the python scripts for training together with a README where it is explained how to start the scripts and with which parameters.
