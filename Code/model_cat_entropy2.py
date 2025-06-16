import numpy as np

import tensorflow as tf
np.random.seed(0)
from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.utils import plot_model
import keras.backend as K

np.random.seed(1)

from tokenizers import Tokenizer
from math import floor, ceil
import json
import os
import random

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def find_closest_embedding(predicted, embedding_weights, id2token):
    tokens = list()    
    
    for token_embedding in predicted:
        tokens.append(np.argmax(vector_matrix_cos_sim(token_embedding, embedding_weights)))
    
    tokens = [id2token[tokenid] for tokenid in tokens]
    return tokens

def vector_vector_cos_sim(a, b):
    "" "cosine similarity between vectors" ""
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))


def vector_matrix_cos_sim(a, B):
    "" "cosine similarity between vector and matrix" ""
    return np.dot(B, a) / (np.linalg.norm(a) * np.linalg.norm(B, axis=1))

def index_of_vec(vec, arr, thresh):
    global token2id
    ind=0
    for ind,vec2 in enumerate(arr):
        temp = vector_vector_cos_sim(vec, vec2)
        if temp >= thresh or temp <= -thresh:
            return ind
    return token2id['<UNK>']


# Even though we define the encoder and decoder models we still need to dynamically provide the decoder_input_data as follows:

# it begins with a special symbol start
# it will continue with an input created by the decoder at previous time step
# in other words, decoder's output at time step t will be used decoder's input at time step t+1


def decode_sequence(batch_input, tokenizer, id2token, token2id):
    global encoder_model, decoder_model, embedding_weights
    
    # Encode the input as state vectors.
    states_value = encoder_model.predict(batch_input, verbose = 0)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, max_deco_input_len, embedding_weights.shape[1])) # one batch, sent length, word embedding dim
    # Populate the first token of target sequence with the start token.
    target_seq[0, 0, :] = embedding_weights[token2id['<START>'],:]

    decoded_seq = list()
    stop_condition = False
    
    # Sampling loop for the length of the sequence
    for j in range(max_deco_input_len-1):
        if stop_condition:
            break
        # decode the input to a token/output prediction + required states for context vector Update input states (context vector) 
        # with the outputed states
        output, h, c =  decoder_model.predict([target_seq] + states_value, verbose = 0)
        
        # convert the token/output prediction to a token/output
        sampled_token_embedding = output[0,j,:]
        
        # add the predicted token/output to output sequence
        decoded_seq.append(sampled_token_embedding)
        
        # Exit condition: find <END> token.
        if vector_vector_cos_sim(sampled_token_embedding, embedding_weights[token2id['<END>'],:]) >= 0.9:
            stop_condition = True

        # Update the input target sequence with the predicted token/output 
        target_seq[:, j + 1, :] = sampled_token_embedding
        
        states_value = [h, c]
        
    return decoded_seq


def decode(decoded_seq):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    encoding -- a list containing the tokens 
    
    Returns: 
    decoding -- a string summerizing all tokens.
    '''
    decoding = ''
    for token in decoded_seq:
        if token == '</w>':
            decoding = decoding + ' '
        else:
            decoding = decoding + token
    return decoding

def calculating_combined_embeddings(embedding_weights, batch_input, tokenizer, token2id):
    '''
    Creates a tensor of shape (batch size, sent length, word embedding dim) as input to the encoder 
    by calculating the combined effect of tokens' weights as a whole word
    
    Arguments:
    embedding_weights -- a 2D array of embbedings
    batch_input -- a list of lists of tokens for input sentences
    tokenizer -- the BPE tokenizer to tokenize batch input
    token2id -- a dictionary the translates tokens to their respective ids
    
    Returns: 
    batch_input_tensor -- a tensor of shape.
    '''
    batch_input_tensor = np.ndarray((len(batch_input), len(batch_input[0]), embedding_weights.shape[1])) # batch size, sent length, word embedding dim
    sent_embedding = np.zeros((len(batch_input[0]), embedding_weights.shape[1]))
    word_embedding = 0
    
    for i, sent in enumerate(batch_input):
        sent_embedding.fill(0)
        for j, word in enumerate(sent):
            word_encoding = tokenizer.encode(word).tokens
            word_embedding = 0
            for k in range(len(word_encoding)):
                try:
                    word_embedding = word_embedding + embedding_weights[token2id[word_encoding[k]]][:]
                except:
                    word_embedding = word_embedding + embedding_weights[token2id['<UNK>'],:]
            word_embedding = word_embedding / len(word_encoding) if len(word_encoding) != 0 else embedding_weights[token2id['<UNK>'],:]
            word_embedding.resize((1,embedding_weights.shape[1]))
            np.put_along_axis(arr=sent_embedding, indices=np.full((1,embedding_weights.shape[1]),j), values=word_embedding, axis=0)
        np.put_along_axis(arr=batch_input_tensor, indices=np.full((1,len(batch_input[0]),embedding_weights.shape[1]),i), values=sent_embedding, axis=0)
    return batch_input_tensor

def modified_tanh(x):
    return K.tanh(x)*2

with open('D:\\Project Files\\encoder_input_data.json') as f:
   encoder_input_data = json.load(f)

with open('D:\\Project Files\\decoder_input_data.json') as f:
   decoder_input_data = json.load(f)

with open('D:\\Project Files\\decoder_output_data.json') as f:
   decoder_output_data = json.load(f)

tokenizer = Tokenizer.from_file("D:\\Project Files\\tokenizer_trained.json")

embedding_weights = np.load("D:\\Project Files\\embedding_weights.npy")

token2id = tokenizer.get_vocab()
id2token = {v:k for k, v in token2id.items()}
unique_tokens = len(tokenizer.get_vocab())
data_size = len(encoder_input_data)
max_enco_input_len = len(encoder_input_data[0])
max_deco_input_len = len(decoder_input_data[0])
max_deco_output_len = len(decoder_output_data[0])
embedding_dim = embedding_weights.shape[1]
batch_size = 32
cut_percentage = 0.8
cut = floor(data_size*cut_percentage)
n_features = 50


X_Train_enco = encoder_input_data[:cut] + [ ['<PAD>'] * max_enco_input_len for _ in range(batch_size-cut%batch_size)]
X_Test_enco = encoder_input_data[cut:] + [ ['<PAD>'] * max_enco_input_len for _ in range(batch_size-(data_size - cut)%batch_size)]

X_Train_deco = decoder_input_data[:cut] + [ ['<PAD>'] * max_deco_input_len for _ in range(batch_size-cut%batch_size)]

Y_Train_deco = decoder_output_data[:cut] + [ ['<PAD>'] * max_deco_output_len for _ in range(batch_size-cut%batch_size)]
Y_Test_deco = decoder_output_data[cut:] + [ ['<PAD>'] * max_deco_output_len for _ in range(batch_size-(data_size - cut)%batch_size)]
    
# X_enco = list()
# X_deco = list()
# Y_deco = list()
# for batch in range(int(len(X_Train_enco)/batch_size)):
#     X_enco.append(calculating_combined_embeddings(embedding_weights, X_Train_enco[batch:batch+batch_size], tokenizer, token2id))
#     X_deco.append(calculating_combined_embeddings(embedding_weights, X_Train_deco[batch:batch+batch_size], tokenizer, token2id))
#     Y_deco.append(calculating_combined_embeddings(embedding_weights, Y_Train_deco[batch:batch+batch_size], tokenizer, token2id))

# TRAINING WITH TEACHER FORCING
encoder_inputs= Input(shape=(max_enco_input_len,embedding_dim), name='encoder_inputs')
encoder_lstm = LSTM(units=50, return_state=True, name='encoder_lstm')
LSTM_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# We discard `LSTM_outputs` and only keep the other states.
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_deco_input_len, embedding_dim), name='decoder_inputs')
decoder_lstm = LSTM(units=50, return_sequences=True, return_state=True, name='decoder_lstm')

# Set up the decoder, using `context vector` as initial state.
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

#complete the decoder model by adding a Dense layer with Softmax activation function 
#for prediction of the next output
#Dense layer will output one-hot encoded representation
decoder_dense = Dense(embedding_dim, activation=modified_tanh, name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# put together
model_encoder_training = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='model_encoder_training')

opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, clipnorm=5.0)
loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

model_encoder_training.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
model_encoder_training.summary()

model_encoder_training.load_weights('D:\\Project Files\\trained_model_weights.h5')

# tf.keras.utils.plot_model(
#     model_encoder_training,
#     to_file="BP_training_model2.png",
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=True,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=True
# )

# l = list()
# a = list()

# # optimization loop
# for epoch in range(1, 100):
#     loss = 0
#     acc = 0
#     for batch in range(int(len(X_Train_enco)/batch_size)):        
#         temp = model_encoder_training.train_on_batch([X_enco[batch], X_deco[batch]], Y_deco[batch])
        
#         loss = loss + temp[0]
#         acc = acc + temp[1]
    
#     loss = loss / int(len(X_Train_enco)/batch_size)
#     acc = acc / int(len(X_Train_enco)/batch_size)
#     l.append(loss)
#     a.append(acc)
#     print('Epoch:', epoch, 'Loss:', loss, 'Accuracy:', acc)

# TESTING WITHOUT TEACHER FORCING
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(50,), name='encoder_state_h')
decoder_state_input_c = Input(shape=(50,), name='encoder_state_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states, name='model_decoder_testing')
model_encoder_training.summary()

# tf.keras.utils.plot_model(
#     decoder_model,
#     to_file="BP_testing_model.png",
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=True,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=True
# )


# acc = 0 
# test_size = len(X_Test_enco)
# for sample in range(0,test_size):
#     X_enco = calculating_combined_embeddings(embedding_weights, [X_Test_enco[sample]], tokenizer, token2id)
#     Y_deco = calculating_combined_embeddings(embedding_weights, [Y_Test_deco[sample]], tokenizer, token2id)
#     predicted = decode_sequence(X_enco, tokenizer=tokenizer, id2token=id2token,token2id=token2id)
    
#     len_actual = len(predicted)
#     len_desired = Y_Test_deco[sample].index('<PAD>')
    
#     comp = min(len_actual, len_desired)
#     correct = 0
    
#     for i in range(comp):
#         temp = vector_vector_cos_sim(Y_deco[0,i,:], predicted[i])
#         if temp >= 0.6 or temp <= -0.6:
#             correct += 1
    
#     acc += correct/len_desired
#     print(acc)
#     print(sample)

# acc = acc / test_size

test_size = len(X_Test_enco)
for i in range(3):
    sample = random.randint(0,test_size)
    X_enco = calculating_combined_embeddings(embedding_weights, [X_Test_enco[sample]], tokenizer, token2id)
    Y_deco = calculating_combined_embeddings(embedding_weights, [Y_Test_deco[sample]], tokenizer, token2id)
    predicted = decode_sequence(X_enco, tokenizer=tokenizer, id2token=id2token,token2id=token2id)
    predicted = find_closest_embedding(predicted, embedding_weights, id2token)
    print("Example ",i," : \n","Input : ",X_Test_enco[sample],
          "\n\nDesired Output : ",Y_Test_deco[sample],
          "\n\nActual Output : ",predicted,'\n\n')
    