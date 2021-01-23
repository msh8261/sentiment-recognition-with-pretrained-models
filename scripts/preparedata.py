
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import tensorflow as tf
#import os
#import numpy as np
#import pandas as pd
#import tensorflow_hub as hub


def convert_to_dfts(df, labels):
  ds = tf.data.Dataset.from_tensor_slices((
      tf.cast(df, tf.string),
      tf.cast(labels, tf.int32)
      ))
  return ds


def split_dataset_csv(path_csv, split_percent):
    sentences = []
    labels    = []
    with open(path_csv, encoding='utf-8') as fp:
      lines = csv.reader(fp)
      next(lines)
      for row in lines:
        #print(row[1])
        sentences.append(row[0])
        label = 1 if row[1]=='positive' else 0
        labels.append(int(label))
        
    split_percent = split_percent
    split_to      = int(split_percent * len(sentences))
    # Spliting data into 60-40
    train_sentences = sentences[0:split_to]
    val_split     = int(split_percent * len(train_sentences))
    # Further divind 60% training data into 60-40 for validation data
    val_sentences   = train_sentences[val_split:]
    train_sentences = train_sentences[0:val_split]
    test_sentences  = sentences[split_to:]
    train_labels = labels[0:split_to]
    val_labels   = train_labels[val_split:]
    train_labels = train_labels[0:val_split]
    test_labels  = labels[split_to:]

    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels


def tensor_dataset(train_sentences, train_labels, 
    val_sentences, val_labels, test_sentences, test_labels):
    #Creating Dataset for training, validation and testing datatframes.
    train_ds = convert_to_dfts(train_sentences, train_labels)
    val_ds   = convert_to_dfts(val_sentences  , val_labels  )
    test_ds  = convert_to_dfts(test_sentences , test_labels )

    return train_ds, val_ds, test_ds
