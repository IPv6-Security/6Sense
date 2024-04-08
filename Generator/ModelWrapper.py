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
import time
import sys
import math
import shutil

import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

from Generator.NN_models import GeneratorMaskedLSTM
from Generator.cython_print_ips import print_ips
from Generator.DatasetEncoders import AllocationEncoder

from Generator.Sampling import predict_base_function

import config as conf



class ModelBase():
    def __init__(self, 
                 dataset, 
                 PriorLevel, 
                 hps, 
                 *args):
        """
        LSTM Base for ORGAN Generator. 
        """
        ## GENERAL HYPERPARAMETERS
        self.SetHyperparameters(hps)

        # How far into each dataset row to begin generation (often the same as the sequence length)
        self.START_COLUMN = self.SEQUENCE_LENGTH
        # Dimensions of Labels (often the same as the sequence_length/subnet_offset if the labels are the ungenerated prefix)
        self.LABEL_DIMENSION = self.SEQUENCE_LENGTH # 8 # label_dim

        
        # Training Dataset name
        self.DATASET = dataset
        
        # Category Names for Datasets:
        self.CATEGORIES = list(dataset.keys())

        
        # Function used for computing upper bits for samples
        self.UPPER_SEQUENCE_GENERATION_FUNCTION = PriorLevel

        # Pretain:
        self.GEN_BATCH_SIZE = self.BATCH_SIZE*2 #train_data_tensor.shape[0] #sequences_normalized.shape[0]
        
        # Loss
        self.GENERATOR_LOSS = []
        
        self.GENERATOR_PRETRAIN_EPOCHS = 10
        
        ## Distributed Scope
                
        self.CURRENT_MODEL_CATEGORY = self.CATEGORIES[0]
        
        ## IP SPECIFIC METRIC:
        self.IPs_generated = 0
        
        # FUNCTIONS FOR INTILIAZING STRUCTURE       
        ## CREATE DATASETS:
        self.CreateDataset(self.DATASET_ENCODING)
        ## OPTIMIZERS
        
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate = self.GENERATOR_LEARNING_RATE, beta_1=self.BETA)
        ## CREATE GENERATOR
        self.model = self.CreateLSTM()
                
        self.LoadLSTM()
        
    def SetHyperparameters(self, hyperparameters):
        """
        Load Hypterparameters from input and from Default.
        """
        # Sequence Length
        self.SEQUENCE_LENGTH = 16
        if "seq_length" in hyperparameters:
            self.SEQUENCE_LENGTH = hyperparameters["seq_length"]

        # Maximum Length to keep of each dataset row:
        self.MAXIMUM_LENGTH = 32
        if "max_length" in hyperparameters:
            self.MAXIMUM_LENGTH = hyperparameters["max_length"]

        # Neural Network Model
        self.model_function = GeneratorMaskedLSTM
        if "model" in hyperparameters:
            self.model_function = hyperparameters["model"]
            
        # Sampling Function:
        self.predict_generator_lstm = predict_base_function
        if "sampler" in hyperparameters:
            self.predict_generator_lstm = hyperparameters["sampler"]        

        # Dataset Encoder
        self.encoder_class = AllocationEncoder
        if "encoder" in hyperparameters:
            self.encoder_class = hyperparameters["encoder"]

        # Unique Values in the dictionary:
        self.UNIQUE_NYBBLE_VALUES = 16
        if "unique_nybbles" in hyperparameters:
            self.UNIQUE_NYBBLE_VALUES = hyperparameters["unique_nybbles"]

        # If the generator should be loaded from a checkpoint        
        self.LOAD_GENERATOR_FROM_CHECKPOINT = False
        if "preload" in hyperparameters:
            self.LOAD_GENERATOR_FROM_CHECKPOINT = hyperparameters["preload"]

        # If print statements should be triggered
        self.DEBUG = False
        if "debug" in hyperparameters:
            self.DEBUG = hyperparameters["debug"]

        self.BUFFER_SIZE = 3000000
        if "BUFFER_SIZE" in hyperparameters:
            self.BUFFER_SIZE = hyperparameters["BUFFER_SIZE"]

        self.BATCH_SIZE = 3200
        if "BATCH_SIZE" in hyperparameters:
            self.BATCH_SIZE = hyperparameters["BATCH_SIZE"]

        self.SAMPLING_BATCH_SIZE = 20000
        if "SAMPLING_BATCH_SIZE" in hyperparameters:
            self.SAMPLING_BATCH_SIZE = hyperparameters["sampling_batch_size"]

        # The layers of the generator
        self.GENERATOR_LAYERS = [512, 256]
        if "layers" in hyperparameters:
            self.GENERATOR_LAYERS = hyperparameters["layers"]

        # Dropout for the generator
        self.GENERATOR_DROPOUT = 0 # 0.1 or 0.2 are acceptable values (but beware of going too high)
        if "dropout" in hyperparameters:
            self.GENERATOR_DROPOUT = hyperparameters["dropout"]

        # Checkpoint Filepaths
        self.GENERATOR_CHECKPOINT = {"all":"lstm_organ"}
        if "checkpoint" in hyperparameters:
            self.GENERATOR_CHECKPOINT = hyperparameters["checkpoint"]
        else:
            raise ValueError("No Value Passed for Generator Checkpoint.")
        self.GENERATOR_CHECKPOINT_FILEPATH = {}
        self.GENERATOR_CHECKPOINT_INTERMEDIATE = {}

        for c in self.GENERATOR_CHECKPOINT:
            self.GENERATOR_CHECKPOINT_FILEPATH[c] = self.GENERATOR_CHECKPOINT[c] + ".hdf5"
            self.GENERATOR_CHECKPOINT_INTERMEDIATE[c] = self.GENERATOR_CHECKPOINT[c] + "_intermediate.hdf5"

        # Generator Learning Rate:
        self.GENERATOR_LEARNING_RATE = 3e-4
        if "lr" in hyperparameters:
            self.GENERATOR_LEARNING_RATE = hyperparameters["lr"]
        # Beta:
        self.BETA = 0.7
        if "beta" in hyperparameters:
            self.BETA = hyperparameters["beta"]
        ## CREATE DATASETS:
        self.DATASET_ENCODING = 0
        if "dataset_encoding_bit" in hyperparameters:
            self.DATASET_ENCODING = hyperparameters["dataset_encoding_bit"]
            
        # Validation Split:
        self.VALIDATION_SPLIT = 0
        if "validation_split" in hyperparameters:
            self.VALIDATION_SPLIT = hyperparameters["validation_split"]


    def SaveInterWeights(self, c):
        """
        Save weights of individual models for switching between models
        """
        self.model.save_weights(self.GENERATOR_CHECKPOINT_INTERMEDIATE[c])
        self.CURRENT_MODEL_CATEGORY = c
        
    def LoadInterWeights(self, c):
        """
        Load weights of intermediate model when switching back to it.
        """
        self.model.load_weights(self.GENERATOR_CHECKPOINT_INTERMEDIATE[c])
        self.CURRENT_MODEL_CATEGORY = c
    
    
    def CreateDataset(self, encoding):
        self.SEQUENCE_ENCODER = self.encoder_class(sequence_length=self.SEQUENCE_LENGTH, unique_nybble_values=self.UNIQUE_NYBBLE_VALUES)
        self.SEQUENCE_ENCODER.CreateSequences(self.DATASET, encoding=encoding)
        
        self.sequences_normalized, self.labels_normalized, self.sequence_prefixes = self.SEQUENCE_ENCODER.getDatasets()
        self.sequence_tf_dataset, self.validation_data = self.SEQUENCE_ENCODER.distributeDataset("", self.BUFFER_SIZE, self.BATCH_SIZE, self.VALIDATION_SPLIT)
    
    
    def CreateLSTM(self):

        generator = self.model_function(self.GENERATOR_LAYERS, 
                                  (self.sequences_normalized[self.CATEGORIES[0]].shape[1],
                                  self.sequences_normalized[self.CATEGORIES[0]].shape[2]),
                                  self.GENERATOR_DROPOUT,
                                  self.labels_normalized[self.CATEGORIES[0]].shape[1])        
        # Compile Model
        generator.compile(loss='categorical_crossentropy', 
                          optimizer=self.generator_optimizer, 
                          metrics=['accuracy'])
        return generator
    
    
    def LoadLSTM(self):
        """
        Load Generator from Checkpoint
        """
        # Initiate Models:
        sequence = self.generate_sequence(100)        
        self.GENERATOR_LOSS = {}
        for c in self.GENERATOR_CHECKPOINT_FILEPATH:
            if self.LOAD_GENERATOR_FROM_CHECKPOINT == True:
                self.model.load_weights(self.GENERATOR_CHECKPOINT_FILEPATH[c])
                print("Loaded Generator From Checkpoint File!")
                
            self.SaveInterWeights(c)
            self.GENERATOR_LOSS[c] = []
            
    def LoadFromSeparatePath(self, path):
        self.model.load_weights(path)
        print("Loaded Generator From Checkpoint File!")
    
    
    def TrainLSTM(self, epochs, plot=True, train_test_split=0.15, save_iteration=25, name="LSTM"):
        """
        MLE Pretrain the Generator
        """
        # TODO: MLE {Pretraining of the Generator on the Seed dataset}
        self.GENERATOR_PRETRAIN_EPOCHS = epochs
        
        
        for c in self.CATEGORIES:
            if self.LOAD_GENERATOR_FROM_CHECKPOINT == False:
                
                self.LoadInterWeights(c)
                
                # Add Model checkpoint - for intermediate models
                checkpoint = ModelCheckpoint(self.GENERATOR_CHECKPOINT_FILEPATH[c].split('.')[0] + "-{epoch:02d}.hdf5", 
                                             monitor='loss',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='min')
                desired_callbacks = [checkpoint, tf.keras.callbacks.History()]  #, tensorboard_callback] #, self.tensorboard_callback, self.metric_callback]
                
                print("Dataset Shape: ", self.sequences_normalized[c].shape)
                
                if plot == True:
                    loss = self.model.fit(self.sequence_tf_dataset[c], 
                                      epochs=self.GENERATOR_PRETRAIN_EPOCHS, 
                                      callbacks=desired_callbacks, 
                                      steps_per_epoch=math.ceil(self.sequences_normalized[c].shape[0]/self.BATCH_SIZE),
                                      validation_data = self.validation_data[c])

                else:
                    loss = self.model.fit(self.sequence_tf_dataset[c], 
                                      epochs=self.GENERATOR_PRETRAIN_EPOCHS, 
                                      callbacks=desired_callbacks, 
                                      steps_per_epoch=math.ceil(self.sequences_normalized[c].shape[0]/self.BATCH_SIZE))
                    
                
                self.GENERATOR_LOSS[c] += loss.history['loss']

                
                # Save to Final Checkpoint File
                self.model.save_weights(self.GENERATOR_CHECKPOINT_FILEPATH[c])
                
                
                print(loss.history.keys())
                
                if plot == True:
                    plt.rcParams.update({'font.size': 26})
                    plt.figure(figsize=(20, 10))
                    plt.plot(range(0, epochs), loss.history['loss'], label='Training Loss')
                    plt.plot(range(0, epochs), loss.history['val_loss'], label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title("Learning Curve for: " + name)
                    plt.legend()
                    plt.grid()
                    plt.show()
                                
                self.SaveInterWeights(c)
                
    def TrainOnData(self, epochs, data):
        pass
                
    
    def generate(self, number_of_samples, format_ips=True, debug=False):
        """
        Generate formatted IPs
        """
        print("GENERATING WITH PREFIX INPUT: ", number_of_samples)
        patterns = self.generate_sequence(number_of_samples)
        patterns = patterns*self.UNIQUE_NYBBLE_VALUES
                
        # TODO: Convert IPs
        if format_ips == True:
            full_ips = print_ips(patterns.astype(int))            
            return full_ips
        else:
            return patterns
        
    def sample(self, numberToGenerate, plot=False, total=False):
        """
        Sample from model.
        """
        single_sequences = self.generate_sequence(numberToGenerate)
        return single_sequences
    
    def update(self, hits, *args):
        """
        Spot for updating the model during training is needed. 
        """
        pass
        
    def generate_sequence(self, number_of_sequences):
        """
        Generically Generate a Sequence
        """
        sequences = np.zeros((number_of_sequences, self.MAXIMUM_LENGTH - self.START_COLUMN + self.SEQUENCE_LENGTH))
        if self.UPPER_SEQUENCE_GENERATION_FUNCTION is None:
            sequences[:,0:self.SEQUENCE_LENGTH] = self.sequence_prefixes[self.CURRENT_MODEL_CATEGORY][np.random.randint(0, self.sequence_prefixes[self.CURRENT_MODEL_CATEGORY].shape[0]-1, size=number_of_sequences)]
            
            self.CURRENT_UPPER_SEQUENCE_LENGTHS = (np.ones((sequences.shape[0]))*self.SEQUENCE_LENGTH).astype(int)
            
        elif self.UPPER_SEQUENCE_GENERATION_FUNCTION == "SEQUENCE_MODEL":
            sequences = self.SEQUENCE_ENCODER.generateUpper(number_of_sequences, self.CURRENT_MODEL_CATEGORY)
            self.CURRENT_UPPER_SEQUENCE_LENGTHS = self.SEQUENCE_ENCODER.CURRENT_UPPER_SEQUENCE_LENGTHS
        else:
            upper = self.UPPER_SEQUENCE_GENERATION_FUNCTION(number_of_sequences, self.CURRENT_MODEL_CATEGORY)
            
            # At some point fix this by standardizing dataset representations. 
            if self.SEQUENCE_LENGTH != 16:
                mask = (upper==0)
                #print(mask)
                self.CURRENT_UPPER_SEQUENCE_LENGTHS = np.where(mask.any(1), mask.argmax(1), upper.shape[1])
            else:
                self.CURRENT_UPPER_SEQUENCE_LENGTHS = (np.ones((sequences.shape[0]))*self.SEQUENCE_LENGTH).astype(int)

            sequences[:, 0:upper.shape[1]] = upper
                    
        return self._generation_core(sequences, predict_lstm = self.predict_generator_lstm)


    def generate_from_input(self, sequences, current_upper_sequence_lengths):
        self.CURRENT_UPPER_SEQUENCE_LENGTHS = current_upper_sequence_lengths
        return self._generation_core(sequences, predict_lstm = self.predict_generator_lstm)
    
    def _generation_core(self, sequences, predict_lstm, t_val = 0, batch_size = 0):
        """
        Main Generation Function
            sequences: The initial sequences to generate from as a numpy array.
            predict_lstm: The compiled prediction function to use.
            beta: Whether to use the beta generator (i.e. for rollout)
            t_val: The nybble value to begin generation at
            batch_size: batch_size = 12500 optimial
        """
        ### SHAPE INTO CORRECT ENCODING

        batch_size=self.SAMPLING_BATCH_SIZE

        starting_sequences, starting_sequences_actual, tf_sequences = self.SEQUENCE_ENCODER.formatForGeneration(sequences, t_val, batch_size)

        t1 = time.time()
        i = 0
        
        while i < starting_sequences.shape[0]:
            tf_sequences2 = tf_sequences[i:batch_size+i, :]    
            for t in range(t_val, self.MAXIMUM_LENGTH - self.START_COLUMN):
                lstm_input = self.SEQUENCE_ENCODER.createGeneratedColumn(tf_sequences2, t)
                if self.DEBUG:
                    print(tf_sequences2.shape, t)
                new_column =  predict_lstm(self.model, lstm_input)
                tf_sequences2 = self.SEQUENCE_ENCODER.formatGeneratedColumn(new_column, tf_sequences2, t+self.START_COLUMN) 
            if i == 0:
                sequences_to_return = tf_sequences2
            else:
                sequences_to_return = tf.concat([sequences_to_return, tf_sequences2], 0)
            i += batch_size

        sequences_to_return = sequences_to_return[0:starting_sequences_actual.shape[0]]

        t2 = time.time()
        if self.DEBUG:
            print(sequences)
            print("Total Time: ", t2-t1)
            print("IPS/Second: ", starting_sequences.shape[0]/(t2-t1))


        ### RESHAPE INTO NUMPY ARRAY OF IP NUMBERS
        returnable_sequences = self.SEQUENCE_ENCODER.formatGenerated(sequences_to_return, sequences)
        
        return returnable_sequences

