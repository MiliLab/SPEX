import torch
from llava.train.train import train
import os
os.environ["WANDB_MODE"]="disabled"
if __name__ == "__main__":
    train()
