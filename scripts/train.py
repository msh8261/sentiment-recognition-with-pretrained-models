
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scripts.nlpfuncs import train_evaluate_pretrained_models
from scripts.preparedata import convert_to_dfts, split_dataset_csv, tensor_dataset


def train_models():

	# Embedding Layer
	embedding1 = "https://tfhub.dev/google/nnlm-en-dim128/2"
	embedding2 = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"
	embedding3 = "https://tfhub.dev/google/nnlm-en-dim50/2"
	embedding4 = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
	embedding5 = "https://tfhub.dev/google/universal-sentence-encoder/4"

	save_model_name1  = 'my_model_nnlm-en-dim128.h5'
	save_model_name2  = 'my_model_nnlm-en-dim128-with-normalization.h5'
	save_model_name3  = 'my_model_nnlm-en-dim50.h5'
	save_model_name4  = 'my_model_gnews-swivel-20dim.h5'
	save_model_name5  = 'my_model_universal-sentence-encoder.h5'

	results = {}
	batch_size = 256
	shuffle = 10000
	epochs = 15

	path_csv = "data/movie_reviews.csv"
	split_percent = 0.7

	train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels = split_dataset_csv(path_csv, split_percent)

	train_ds, val_ds, test_ds = tensor_dataset(train_sentences, train_labels, 
											val_sentences, val_labels, 
											test_sentences, test_labels)

	#results["nnlm-en-dim128"] = train_evaluate_pretrained_models(embedding1, save_model_name1, shuffle, epochs, batch_size, train_ds, val_ds, test_ds)
	results["nnlm-en-dim128-with-normalization"] = train_evaluate_pretrained_models(embedding2, save_model_name2, shuffle, epochs, batch_size, train_ds, val_ds, test_ds)
	results["nnlm-en-dim50"] = train_evaluate_pretrained_models(embedding3, save_model_name3, shuffle, epochs, batch_size, train_ds, val_ds, test_ds)
	results["gnews-swivel-20dim"] = train_evaluate_pretrained_models(embedding4, save_model_name4, shuffle, epochs, batch_size, train_ds, val_ds, test_ds)
	#results["universal-sentence-encoder"] = train_evaluate_pretrained_models(embedding5, save_model_name5, shuffle, epochs, batch_size, train_ds, val_ds, test_ds)


