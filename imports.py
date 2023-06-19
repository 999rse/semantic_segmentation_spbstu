import os
import glob
import zipfile
import pickle

import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, CenterCrop
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(4)