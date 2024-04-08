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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#@tf.function(experimental_relax_shapes=True)


@tf.function(jit_compile=True)
def predict_base_function_faster(model, lstm_input):
    """
    Compilable Tensorflow prediction function
    """
    prediction = model(lstm_input)
    new_column = tf.random.categorical(tf.math.log(prediction), num_samples=1)
    new_column = tf.reshape(new_column, [new_column.shape[0], new_column.shape[1], 1])
    new_column = tf.cast(new_column, dtype=tf.double)
    return new_column

@tf.function(experimental_relax_shapes=True)
def predict_base_function(model, lstm_input, **args):
    """
    Compilable Tensorflow prediction function
    """
    prediction = model(lstm_input)
    new_column = tf.random.categorical(tf.math.log(prediction), num_samples=1)
    new_column = tf.reshape(new_column, [new_column.shape[0], new_column.shape[1], 1])
    new_column = tf.cast(new_column, dtype=tf.double)
    return new_column
