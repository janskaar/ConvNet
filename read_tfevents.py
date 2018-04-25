import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np
import matplotlib.pyplot as plt

events_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/300ms/g/first/')

ea = event_accumulator.EventAccumulator(events_path)
ea.Reload()
