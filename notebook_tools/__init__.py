import pickle
import os
import pandas as pd
import numpy as np

from IPython.display import display, HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from glob import glob
import torch

import matplotlib.cm as cm

import random

mpl.rcParams['figure.dpi'] = 100
pd.set_option('display.max_columns', 500)

def eig_sorted(A):
    eigvalues, eigenvectors = torch.eig(A, eigenvectors=True)
    sorted_values, sorted_indices = eigvalues[:,0].sort(descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigvalues[sorted_indices, :], eigenvectors
