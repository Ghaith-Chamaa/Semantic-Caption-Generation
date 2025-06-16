import os

from tokenizers import Tokenizer

from IPython.display import SVG

import json

import tensorflow as tf

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams 
from keras.layers import *
from keras.layers.core import Dense, Reshape
from tensorflow.keras.layers import Embedding
from keras.models import Model,Sequential
from keras.utils.vis_utils import model_to_dot

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


with open('D:\\Project Files\\doc_data.json') as f:
   doc_data = json.load(f)

tokenizer = Tokenizer.from_file("D:\\Project Files\\tokenizer_trained.json")

token2id = tokenizer.get_vocab()
id2token = {v:k for k, v in token2id.items()}

vocab_size = len(token2id) + 1

wids = [[token2id[token] for token in tokenizer.encode(rel).tokens] for rel in doc_data]

skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=4) for wid in wids]

# pairs, labels = skip_grams[0][0], skip_grams[0][1]
# for i in range(10):
#     print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
#           id2token[pairs[i][0]], pairs[i][0], 
#           id2token[pairs[i][1]], pairs[i][1], 
#           labels[i]))
    

#### The input is targeted words, and context word pair means we need to process two inputs.
#### This input is passed to a separate embedding layer to get word embedding for target and context words

embed_size = 200
word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size,embeddings_initializer="glorot_uniform",input_length=1))
word_model.add(Reshape((embed_size, )))

context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size,embeddings_initializer="glorot_uniform",input_length=1))
context_model.add(Reshape((embed_size,)))

merged_output = add([word_model.output, context_model.output])  

model_combined = Sequential()
model_combined.add(Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid"))

final_model = Model([word_model.input, context_model.input], model_combined(merged_output))
final_model.compile(loss="mean_squared_error", optimizer="adam")

final_model.summary()
final_model.trainable = True

### SVG : Scalable Vector Graphics
# SVG(model_to_dot(final_model, show_shapes=True, show_layer_names=False, 
#                  rankdir='TB').create(prog='dot', format='svg'))
for epoch in range(1, 50):
    loss = 0
    for i, elem in enumerate(skip_grams):
        pair_first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        pair_second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [pair_first_elem, pair_second_elem]
        Y = labels
        if i % 10000 == 0:
            print('Processed {} (skip_first, skip_second, relevance) pairs'.format(i))
        loss += final_model.train_on_batch(X,Y)  

    print('Epoch:', epoch, 'Loss:', loss)
    

word_embed_layer = word_model.layers[0]
weights = word_embed_layer.get_weights()[0][1:]

distance_matrix = euclidean_distances(weights)
print(distance_matrix.shape)

similar_words = {search_term: [id2token[idx] for idx in distance_matrix[token2id[search_term]-1].argsort()[1:6]+1] 
                   for search_term in ['toilet', 'bathroom','cow', 'phone', 'computer']}

similar_words

# with open("D:\\Project Files\\embedding_weights.json", 'w') as f:
#     json.dump(weights, f)