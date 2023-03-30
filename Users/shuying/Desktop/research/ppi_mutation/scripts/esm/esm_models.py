import torch
from torch.utils.data import Dataset
from os import path
import sys
import esm
sys.path.append((path.dirname(path.dirname(path.dirname( path.abspath(__file__))))))
print(path.dirname(path.dirname(path.dirname( path.abspath(__file__)))))
from scripts.utils import *
import pandas as pd


