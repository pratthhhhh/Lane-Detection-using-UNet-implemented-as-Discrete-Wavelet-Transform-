import os
import random

import cv2
import numpy as np
import glob
import tensorflow as tf
import keras
from keras.utils import normalize
from keras.metrics import MeanIoU
from keras import layers
from keras.models import Model
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
