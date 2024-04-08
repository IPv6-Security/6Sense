"""
Copyright 2024 Georgia Institute of Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 """

import os
import sys
import random

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

"""
BEGIN MODEL FUNCTIONS -----------------------------------------------------------------------------------------------
"""
# Generators
def GeneratorMaskedLSTM(generator_layers, input_shape, dropout, label_shape):
    """
    Create an LSTM Generator from Input Parameters.
        generator_layers: List of Layer sizes for the generator.
        input_shape: A tuple of the input shapes to the LSTM.
        dropout: The dropout value to use in the network.
        label_shape: The shape of the one hot encoded labels.
    """
    generator = Sequential()
    generator.add(layers.InputLayer(input_shape=input_shape))
    generator.add(layers.Masking(mask_value=0.0))
    # Add layers
    generator.add(layers.LSTM(generator_layers[0], 
                              return_sequences=True))
    generator.add(layers.Dropout(dropout))

    for r in range(1, len(generator_layers)):
        if r < len(generator_layers)-1:
            return_sequences = True
        else:
            return_sequences = False

        generator.add(layers.LSTM(generator_layers[r], return_sequences=return_sequences))
        generator.add(layers.Dropout(dropout))

    generator.add(layers.Dense(label_shape, 
                               activation='softmax'))
    
    return generator

