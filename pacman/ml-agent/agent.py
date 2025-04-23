import torch #pytorch
import random
import numpy as np #numpy
from collections import deque #data structure to store memory
#from game import SnakeGameAI, Direction, Point #importing the game created in step 1
from model import Linear_QNet, QTrainer #importing the neural net from step 2
from helper import plot #importing the plotter from step 2

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate