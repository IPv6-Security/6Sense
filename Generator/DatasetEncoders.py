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

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from Generator.cython_print_ips import print_ips
from iplibrary import AS_Processor

import config as conf




class SingleAddressEncoder():
    
    def __init__(self, debug=False, sequence_length=16, unique_nybble_values=16):
        """
        Encode Dataset for input into LSTM. Allow for processing dataset later.
        
        """
        self.SEQUENCE_LENGTH = sequence_length
        self.UNIQUE_NYBBLE_VALUES = unique_nybble_values
        self.DEBUG = debug
        self.DATASET = {}
        self.offset = 0
    
    def CreateSequences(self, datasets, encoding=0):
        """
        Take in Dataset and Create Sequences from it.
        """
        self.DATASET = datasets
        self.DATASET_ENCODING = encoding
        
        self.labels_normalized = {}
        self.sequence_prefixes = {}
        self.sequences_normalized = {}
        self.sequence_tf_dataset = {}
        for c in self.DATASET:
            if self.DEBUG:
                sequence_time_1 = time.time()

            sequence_dataset = self.DATASET[c]
            dataset_size = sequence_dataset.shape[0] * (sequence_dataset.shape[1] - self.SEQUENCE_LENGTH)
            if self.DEBUG:
                print("Dataset Size:", dataset_size)

            # Create Training Sequences:

            sequences = np.zeros((dataset_size, self.SEQUENCE_LENGTH))
            labels = np.zeros((dataset_size, 1))
            counter = 0
            for row in sequence_dataset:
                for x in range(0, row.size-self.SEQUENCE_LENGTH):
                    sequences[counter] = row[x:x+self.SEQUENCE_LENGTH]
                    labels[counter] = row[x+self.SEQUENCE_LENGTH]
                    counter += 1

            ### Normalize
            sequences_normalized = (sequences+1)/float(self.UNIQUE_NYBBLE_VALUES)

            ### Apply Encoding
            individual_length = 1

            ### Shuffle Sequences and Labels:
            sequences_normalized, labels = shuffle(sequences_normalized, labels, random_state=0)
            self.sequences_normalized[c] = np.reshape(sequences_normalized, (dataset_size, self.SEQUENCE_LENGTH, individual_length))

            # Create Testing Sequence Prefixes:
            sequence_prefixes = sequence_dataset[:, 0:self.SEQUENCE_LENGTH]
            self.sequence_prefixes[c] = sequence_prefixes/float(self.UNIQUE_NYBBLE_VALUES)
            
            if self.DEBUG:
                print("Sequence Prefixes: ", self.sequence_prefixes[c][0:10])


            if self.DEBUG:
                print("Sequences: ", sequences)
                print("Sequences Normalized: ", self.sequences_normalized[c])
                print("Sequence Labels: ", labels)

            # Create One Hot Encoded Labels
            arr_2 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16]]
            
            labels2 = np.concatenate((labels, arr_2), axis=0)

            encoder = OneHotEncoder().fit(labels2)
            self.labels_normalized[c] = encoder.transform(labels2).toarray()[0:labels.shape[0]]

            if self.DEBUG:
                print("Normalized Labels: ", self.labels_normalized[c][100:110])
                sequence_time_2 = time.time()
                print("Dataset Creation Time: ", sequence_time_2 - sequence_time_1)

                

    def distributeDataset(self, DISTRIBUTED_STRATEGY, BUFFER_SIZE = 100000, BATCH_SIZE=1000, VALIDATION_SPLIT=0):
        self.validation_data = {}
        for c in self.DATASET:
            # Try Converting the Dataset to a tf.dataset object for speed
            self.sequence_tf_dataset_single = tf.data.Dataset.from_tensor_slices((self.sequences_normalized[c], 
                                                                                  self.labels_normalized[c]))
            
            dataset_size = self.sequences_normalized[c].shape[0]
            self.sequence_tf_dataset_single = self.sequence_tf_dataset_single.shuffle(dataset_size) #.batch(BATCH_SIZE).repeat()

            self.validation_data[c] = self.sequence_tf_dataset_single.take(int(VALIDATION_SPLIT*dataset_size)).batch(BATCH_SIZE)
            self.sequence_tf_dataset_single_2 = self.sequence_tf_dataset_single.skip(int(VALIDATION_SPLIT*dataset_size)).batch(BATCH_SIZE).repeat()
            
            self.sequence_tf_dataset[c] = self.sequence_tf_dataset_single_2
        
        return self.sequence_tf_dataset, self.validation_data

    
    def formatForGeneration(self, sequences, t_val, batch_size):
        """
        Format input sequences for Generation.
        """
        if self.DATASET_ENCODING == 0:
            starting_sequences_actual = sequences[:, :t_val+self.SEQUENCE_LENGTH]
            starting_sequences = np.zeros((batch_size*(int(starting_sequences_actual.shape[0]/batch_size) + 1), 
                                          starting_sequences_actual.shape[1]))
            starting_sequences[:starting_sequences_actual.shape[0],
                               :starting_sequences_actual.shape[1]] = starting_sequences_actual
            tf_sequences = tf.convert_to_tensor(starting_sequences)
            tf_sequences = tf.reshape(tf_sequences, [starting_sequences.shape[0], 
                                                     starting_sequences.shape[1], 
                                                     1])
        elif self.DATASET_ENCODING == 1:
            starting_sequences_actual = sequences[:, :t_val+self.SEQUENCE_LENGTH]
            starting_sequences = np.zeros((batch_size*(int(starting_sequences_actual.shape[0]/batch_size) + 1), 
                                          starting_sequences_actual.shape[1], 2))
            starting_sequences[:starting_sequences_actual.shape[0],
                               :starting_sequences_actual.shape[1], 0] = starting_sequences_actual
            a = np.arange(starting_sequences.shape[1]).reshape(1, starting_sequences.shape[1])
            b = np.repeat(a, starting_sequences_actual.shape[0], axis=0)
            starting_sequences[:starting_sequences_actual.shape[0],
                               :starting_sequences_actual.shape[1], 1] = b
            tf_sequences = tf.convert_to_tensor(starting_sequences)
            tf_sequences = tf.reshape(tf_sequences, [starting_sequences.shape[0], 
                                                     starting_sequences.shape[1], 
                                                     starting_sequences.shape[2]])
        
        return (starting_sequences, starting_sequences_actual, tf_sequences)
    
    def createGeneratedColumn(self, tf_sequences2, t):
        """
        Format LSTM input per t value.
        """
        return tf_sequences2[:, t:t+self.SEQUENCE_LENGTH]
    
    def formatGeneratedColumn(self, new, original, ip_step):
        """
        Format output column of generation
        """
        new = tf.divide(new, self.UNIQUE_NYBBLE_VALUES)
        if self.DATASET_ENCODING == 0:
            return tf.concat([original, new], 1)
        elif self.DATASET_ENCODING == 1:
            ones_column = tf.ones([original.shape[0], 1, 1])
            ones_column = tf.math.scalar_mul(ip_step, ones_column)
            ones_column = tf.cast(ones_column, dtype=tf.double) 
            new_column = tf.concat([new, ones_column], 2)
            return tf.concat([original, new], 1)
    
    def formatGenerated(self, sequences_to_return, sequences):
        """
        Format output generated addresses.
        """
        if self.DATASET_ENCODING == 0:
            # NO ENCODING
            returnable_sequences = np.reshape(sequences_to_return.numpy(), (sequences.shape[0], sequences.shape[1]))
        elif self.DATASET_ENCODING == 1:
            # POSITION VECTOR
            returnable_sequences = sequences_to_return.numpy()
            returnable_sequences = returnable_sequences[:, :, 0]
        elif self.DATASET_ENCODING == 2:
            # ONE HOT
            returnable_sequences = sequences_to_return.numpy()
            returnable_sequences = sequences_to_return.sum(axis=2)
        
        return returnable_sequences
    
    def getDatasets(self):
        return (self.sequences_normalized, self.labels_normalized, self.sequence_prefixes)
    
    
    
class PaddedAddressEncoder(SingleAddressEncoder):
    def __init__(self, debug=False, sequence_length=16, unique_nybble_values=16):
        super().__init__(debug=debug, sequence_length=sequence_length, unique_nybble_values=unique_nybble_values)
        self.offset = 0
        
    def CreateSequences(self, datasets, encoding=0):
        """
        Take in Dataset and Create Sequences from it.
        """
        self.DATASET = datasets
        
        self.labels_normalized = {}
        self.sequence_prefixes = {}
        self.sequences_normalized = {}
        self.sequence_tf_dataset = {}
        for c in self.DATASET:
            if self.DEBUG:
                sequence_time_1 = time.time()

            sequence_dataset = self.DATASET[c] + 1
            dataset_size = sequence_dataset.shape[0] * (sequence_dataset.shape[1] - self.SEQUENCE_LENGTH)
            if self.DEBUG:
                print("Dataset Size:", dataset_size)

            # Create Training Sequences:

            sequences = np.zeros((dataset_size, sequence_dataset.shape[1])) #self.SEQUENCE_LENGTH))
            labels = np.zeros((dataset_size, 1))
            counter = 0
            for row in sequence_dataset:
                for x in range(0, row.size-self.SEQUENCE_LENGTH):
                    sequences[counter, 0:x+self.SEQUENCE_LENGTH] = row[0:x+self.SEQUENCE_LENGTH]
                    labels[counter] = row[x+self.SEQUENCE_LENGTH]
                    counter += 1

            ### Normalize
            sequences_normalized = (sequences)/float(self.UNIQUE_NYBBLE_VALUES)

            ### Apply Encoding - Not Necessary in New Encoding
            individual_length = 1
           
            ### Shuffle Sequences and Labels:
            sequences_normalized, labels = shuffle(sequences_normalized, labels, random_state=0)
            self.sequences_normalized[c] = np.reshape(sequences_normalized, (dataset_size, sequence_dataset.shape[1], individual_length))

            # Create Testing Sequence Prefixes:
            sequence_prefixes = sequence_dataset[:, 0:self.SEQUENCE_LENGTH]
            sequence_prefixes = (sequence_prefixes-1)/float(self.UNIQUE_NYBBLE_VALUES)

            self.sequence_prefixes[c] = sequence_prefixes
            
            if self.DEBUG:
                print("Sequence Prefixes: ", self.sequence_prefixes[c][0:10])


            if self.DEBUG:
                print("Sequences: ", sequences)
                print("Sequences Normalized: ", self.sequences_normalized[c])
                print("Sequence Labels: ", labels)

            # Create One Hot Encoded Labels
            arr_2 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16]]
            
            labels2 = np.concatenate((labels, arr_2), axis=0)

            encoder = OneHotEncoder().fit(labels2)
            self.labels_normalized[c] = encoder.transform(labels2).toarray()[0:labels.shape[0]]

            if self.DEBUG:
                print("Normalized Labels: ", self.labels_normalized[c][100:110])
                sequence_time_2 = time.time()
                print("Dataset Creation Time: ", sequence_time_2 - sequence_time_1)
                
                
    def formatForGeneration(self, sequences, t_val, batch_size):
        """
        Format input sequences for Generation.
        """
        starting_sequences_actual = np.zeros((sequences.shape[0], sequences.shape[1]))
        starting_sequences_actual[:, :t_val + self.SEQUENCE_LENGTH] =  sequences[:, :t_val + self.SEQUENCE_LENGTH] + 1/self.UNIQUE_NYBBLE_VALUES
        starting_sequences = np.zeros((batch_size*(int(starting_sequences_actual.shape[0]/batch_size) + 1), 
                                      starting_sequences_actual.shape[1]))
        starting_sequences[:starting_sequences_actual.shape[0],
                           :sequences.shape[1]] = starting_sequences_actual
        tf_sequences = tf.convert_to_tensor(starting_sequences)
        tf_sequences = tf.reshape(tf_sequences, [starting_sequences.shape[0], 
                                                 starting_sequences.shape[1], 
                                                 1])
        
        return (starting_sequences, starting_sequences_actual, tf_sequences)
    
    def createGeneratedColumn(self, tf_sequences2, t):
        """
        Format LSTM input per t value
        """
        return tf_sequences2
    
    def formatGeneratedColumn(self, new, original, ip_step):
        """
        Format output column of generation
        """
        # Add one to output and scale
        ones_column = tf.ones([new.shape[0], 1, 1])
        ones_column = tf.cast(ones_column, dtype=tf.double)
        new = tf.add(new, ones_column)
        new = tf.divide(new, self.UNIQUE_NYBBLE_VALUES)
        # Set new column in original column.
        unstacked = tf.unstack(original, axis=1)
        unstacked[ip_step] = new[:, :, 0]
        to_return = tf.stack(unstacked, axis=1)
        return to_return
       
    
    def formatGenerated(self, sequences_to_return, sequences):
        """
        Format output generated addresses.
        """
        returnable_sequences = np.reshape(sequences_to_return.numpy(), (sequences.shape[0], sequences.shape[1])) - 1/self.UNIQUE_NYBBLE_VALUES
        
        return returnable_sequences
    
    
    
    
class AllocationEncoder(PaddedAddressEncoder):
    def __init__(self, debug=False, sequence_length=16, unique_nybble_values=16):
        super().__init__(debug=debug, sequence_length=sequence_length, unique_nybble_values=unique_nybble_values)
        self.offset = 1
        
    def CreateSequences(self, datasets, encoding=0, allocation_length =16):
        """
        Take in Dataset and Create Sequences from it.
        """
        self.DATASET = datasets
        self.ALLOCATION_LENGTH = allocation_length
        
        self.labels_normalized = {}
        self.sequence_prefixes = {}
        self.sequences_normalized = {}
        self.sequence_tf_dataset = {}
        for c in self.DATASET:
            if self.DEBUG:
                sequence_time_1 = time.time()

            sequence_dataset = self.DATASET[c] + 1

            ### Calculate Allocation Lengths

            filename_allocation_lengths = conf.HOME_DIRECTORY + conf.DATASET_FILE + "all_ips.allocation_lengths.dataset_encoder.npy"
            if os.path.isfile(filename_allocation_lengths):
                time.sleep(1)
                with open(filename_allocation_lengths, 'rb') as g:
                    self.scaled_allocation_lengths = np.load(g)
            else:
                ip_list = print_ips(self.DATASET[c].astype(int))
                asproc = AS_Processor(ip_list, prefix_filename=conf.UPDATED_PFX_AS_FILE)
                self.scaled_allocation_lengths = np.floor(np.array(asproc.allocation_lengths)/4).astype(int)
                if os.path.isfile(filename_allocation_lengths) == False:
                    with open(filename_allocation_lengths, 'wb') as f:
                        np.save(f, self.scaled_allocation_lengths)

            self.scaled_allocation_lengths[self.scaled_allocation_lengths < self.SEQUENCE_LENGTH] = self.SEQUENCE_LENGTH
            self.scaled_allocation_lengths[self.scaled_allocation_lengths > self.ALLOCATION_LENGTH] = self.SEQUENCE_LENGTH
            
            ### Calculate Dataset Size Based on Allocations
            dataset_size = 0
            for x in self.scaled_allocation_lengths:
                dataset_size += sequence_dataset.shape[1] - x
            
            if self.DEBUG:
                print("Dataset Size:", dataset_size)

            ### Create Training Sequences:
            sequences = np.zeros((dataset_size, sequence_dataset.shape[1]))
            labels = np.zeros((dataset_size, 1))
            counter = 0
            row_counter = 0
            sum_thus_far = 0
            test_counter = 0
            
            for row in sequence_dataset:
                for x in range(0, row.size-self.scaled_allocation_lengths[row_counter]):
                    sequences[counter, 0:x+self.scaled_allocation_lengths[row_counter]] = row[0:x+self.scaled_allocation_lengths[row_counter]]
                    labels[counter] = row[x+self.scaled_allocation_lengths[row_counter]]
                    counter += 1
                row_counter += 1

            ### Normalize
            sequences_normalized = (sequences)/float(self.UNIQUE_NYBBLE_VALUES)            

            ### Apply Encoding - Not Necessary in New Encoding
            individual_length = 1
           
            ### Shuffle Sequences and Labels:
            sequences_normalized, labels = shuffle(sequences_normalized, labels, random_state=0)
            self.sequences_normalized[c] = np.reshape(sequences_normalized, (dataset_size, sequence_dataset.shape[1], individual_length))
            
            # Create Testing Sequence Prefixes:
            sequence_prefixes = np.zeros((sequence_dataset.shape[0], sequence_dataset.shape[1]))
            for d in range(0, sequence_dataset.shape[0]):
                sequence_prefixes[d, :self.scaled_allocation_lengths[d]] = sequence_dataset[d, :self.scaled_allocation_lengths[d]]
            sequence_prefixes = (sequence_prefixes)/float(self.UNIQUE_NYBBLE_VALUES)
            self.sequence_prefixes[c] = sequence_prefixes 
            
            if self.DEBUG:
                print("Sequence Prefixes: ", self.sequence_prefixes[c][0:10])


            if self.DEBUG:
                print("Sequences: ", sequences)
                print("Sequences Normalized: ", self.sequences_normalized[c])
                print("Sequence Labels: ", labels)

            # Create One Hot Encoded Labels
            encoder = OneHotEncoder().fit(labels)
            self.labels_normalized[c] = encoder.transform(labels).toarray()

            if self.DEBUG:
                print("Normalized Labels: ", self.labels_normalized[c][100:110])
                sequence_time_2 = time.time()
                print("Dataset Creation Time: ", sequence_time_2 - sequence_time_1)
                            
            
    def formatForGeneration(self, sequences, t_val, batch_size):
        """
        Format input sequences for Generation.
        """
        
        
        starting_sequences_actual = sequences
        starting_sequences = np.zeros((batch_size*(int(starting_sequences_actual.shape[0]/batch_size) + 1), starting_sequences_actual.shape[1]))
        starting_sequences[:starting_sequences_actual.shape[0],
                           :starting_sequences_actual.shape[1]] = starting_sequences_actual
        tf_sequences = tf.convert_to_tensor(starting_sequences)
        tf_sequences = tf.reshape(tf_sequences, [starting_sequences.shape[0], 
                                                 starting_sequences.shape[1], 
                                                 1])
        
        return (starting_sequences, starting_sequences_actual, tf_sequences)
        
        
    
    def createGeneratedColumn(self, tf_sequences2, t):
        """
        Format LSTM input per t value
        """
        
        return tf_sequences2
        
    
    #@tf.function(experimental_relax_shapes=True)
    def formatGeneratedColumn(self, new, original, ip_step):
        """
        Format output column of generation
        """
        # Add one to output and scale
        ones_column = tf.ones([new.shape[0], 1, 1])
        ones_column = tf.cast(ones_column, dtype=tf.double)
        new = tf.add(new, ones_column)
        new = tf.divide(new, self.UNIQUE_NYBBLE_VALUES)
        
        # Combine with Original
        t = original.numpy()
        update_indices = np.argmin(t, axis=1).flatten()
        if_update = (t[:, -1] == 0).flatten()
        g = t[if_update]
        g[range(np.sum(if_update.astype(int))), update_indices[if_update]] = new.numpy()[if_update, :, 0] #new[if_update]
        t[if_update] = g
        to_return = tf.convert_to_tensor(t) 
        return to_return
       
    
    def formatGenerated(self, sequences_to_return, sequences):
        """
        Format output generated addresses.
        """
        returnable_sequences = np.reshape(sequences_to_return.numpy(), (sequences.shape[0], sequences.shape[1])) - 1/self.UNIQUE_NYBBLE_VALUES
        
        return returnable_sequences
    
    def generateUpper(self, number_of_sequences, current_category):
        """
        Generate Upper Bits
        """
        test_input_sequences = np.zeros((number_of_sequences, self.sequences_normalized[current_category].shape[1]))

        indices = np.random.randint(0, self.sequence_prefixes[current_category].shape[0]-1, size=number_of_sequences)
        test_input_sequences[:,0:self.sequences_normalized[current_category].shape[1]] = self.sequence_prefixes[current_category][indices]
        self.CURRENT_UPPER_SEQUENCE_LENGTHS = np.sum((test_input_sequences != 0).astype(int), axis=1)

        return test_input_sequences
    

        