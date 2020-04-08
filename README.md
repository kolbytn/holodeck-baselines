# Baseline Agents for BYU PCCL Holodeck

This repository provides code for training on task from BYU PCCL's Holodeck simulator (https://holodeck.cs.byu.edu/). It uses pytorch implementations (from https://github.com/astooke/rlpyt) of common RL algorithms such as PPO.

## Installation

Clone repo and install requirements via pip

```pip3 install -r requirements.txt```

## Usage

Run `main.py` to start training on a task. The `scenario` option takes either a scenario associated with a holodeck package you have already installed on your machine or a path to a scenario json file that specifies the `package_name`. In the latter case, the package will be installed automatically. Remember to set `cuda_idx` to an int (eg. 0) if you want to run your model using cuda.

To see other options run: 

```python3 main.py --help```
