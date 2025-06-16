import numpy as np

import tensorflow as tf
np.random.seed(0)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.utils import plot_model
import keras
import keras.backend as K

import pygad
import KerasGA
import pygad.kerasga

np.random.seed(1)

from tokenizers import Tokenizer
import json
from math import floor, ceil
import random
import os
import time

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
batch_fit = 0


def calculating_euclidean_dist(y_true, y_pred):
    acc = 0
    for i, sent in enumerate(y_true):
        acc = acc + np.average(np.abs(matrix_matrix_cos_sim(y_true[i,:,:] , y_pred[i,:,:])))
    return acc / y_true.shape[0]

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

def fitness_func(solution, sol_idx):
    global model_encoder_training, X_Train_enco, X_Train_deco, Y_Train_deco, batch_fit, transfer_learning_model_weights
    
    model_weights_vector = np.concatenate((transfer_learning_model_weights, solution), axis=0)
    
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model_encoder_training, weights_vector=model_weights_vector)

    model_encoder_training.set_weights(weights=model_weights_matrix)

    predictions = model_encoder_training.predict_on_batch([X_Train_enco,X_Train_deco])

    solution_fitness = custom_fit(Y_Train_deco, predictions)
    
    return solution_fitness

def index_of_vec(vec, arr, thresh):
    ind=0
    for ind,vec2 in enumerate(arr):
        temp = vector_vector_cos_sim(vec, vec2)
        if temp >= thresh:
            return ind
    return 1e3

def sim_below_thresh(candidates, sim_thresh, beg, end):
    count = 0
    i = beg
    while i <= end and i < candidates.shape[0]:
        if abs(candidates[i]) < sim_thresh:
            count+=1
        i+=1
    return count

def sim_above_thresh(candidates, sim_thresh, beg, end):
    count = 0
    i = beg
    while i <= end and i < candidates.shape[0]:
        if abs(candidates[i]) > sim_thresh:
            count+=1
        i+=1
    return count

def vector_vector_cos_sim(a, b):
    "" "cosine similarity between vectors" ""
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))

def vector_matrix_cos_sim(a, B):
    "" "cosine similarity between vector and matrix" ""
    return np.dot(B, a) / (np.linalg.norm(a) * np.linalg.norm(B, axis=1))

def matrix_matrix_cos_sim(A, B):
    "" "cosine similarity between matrices" ""
    return  np.dot(A, B.T).diagonal() / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))

def sim_desim(y_true, y_pred, sent_length_reg, desim_reg, sim_thresh, ind_thresh, window_size, token2id, embedding_weights):
    
    batch_fit = np.zeros(y_true.shape[0]) # batch size
    
    pad_embedding = embedding_weights[token2id['<PAD>']][:]
    end_embedding = embedding_weights[token2id['<END>']][:]
    pad_embedding.resize((1,300))
    end_embedding.resize((1,300))
    
    for i, sent in enumerate(y_pred): # for each sent in the batch
        len_actual = min(index_of_vec(pad_embedding, y_pred[i,:,:], ind_thresh), index_of_vec(end_embedding, y_pred[i,:,:], ind_thresh), y_pred.shape[1])
        len_desired = min(index_of_vec(pad_embedding, y_true[i,:,:], ind_thresh), index_of_vec(end_embedding, y_true[i,:,:], ind_thresh), y_true.shape[1])
        sent_fit = np.zeros(len_actual)
        if len_actual >= len_desired:
            X = len_desired - window_size + 1 # num of windows available in desired
            X = X if X > 0 else 1 
            Y = len_actual / X # num of window uses
            Z = len_actual % X # residue
            
            j = 0
            window = 0
            # there are Z windows that will be used ceil(Y) times , and
            # X - Z windows that will be used floor(Y) times
            while j < len_actual:
                
                cross_prod = vector_matrix_cos_sim(y_pred[i,j,:],y_true[i,:,:]) # pred word embedding X all embeddings of the words of the true sent
                
                quotient , residue = divmod(X - Z, 2)
                for window in range(0, quotient):  # half of the minority , i.e. the lower number of uses
                    for usage in range(floor(Y)):
                        if j >= len_actual: break
                        sent_fit[j] = sim_above_thresh(cross_prod, sim_thresh, window, window + window_size - 1) - desim_reg * sim_below_thresh(cross_prod, sim_thresh, window, window + window_size - 1)
                        j = j + 1
                        
                for window2 in range(Z): # the majority , i.e. the bigger number of uses
                    window = window + window2
                    for usage in range(ceil(Y)): 
                        if j >= len_actual: break
                        sent_fit[j] = sim_above_thresh(cross_prod, sim_thresh, window, window + window_size - 1) - desim_reg * sim_below_thresh(cross_prod, sim_thresh, window, window + window_size - 1)
                        j = j + 1
        
                for window3 in range(quotient + residue):  # the other half of the minority
                    window = window + window3
                    for usage in range(floor(Y)):
                        if j >= len_actual: break
                        sent_fit[j] = sim_above_thresh(cross_prod, sim_thresh, window, window + window_size - 1) - desim_reg * sim_below_thresh(cross_prod, sim_thresh, window, window + window_size - 1)
                        j = j + 1
                
        else:    
            X = len_desired - window_size + 1 # num of windows available in desired
            X = X if X > 0 else 1 
            Y = len_actual / X # num of window uses
            Z = (X - len_actual) * 2 # residue
        
            j = 0
            window = 0
            # there are X - Z windows that will be used once (always) , and
            # Z windows that will be used Z / 2 times
            while j < len_actual: # for each word in the sent
                
                cross_prod = vector_matrix_cos_sim(y_pred[i,j,:],y_true[i,:,:]) # pred word embedding X all embeddings of the words of the true sent
                
                quotient , residue = divmod(X - Z, 2)
                for window in range(0, quotient):  # half of the minority , i.e. the lower number of uses
                    if j >= len_actual: break
                    sent_fit[j] = sim_above_thresh(cross_prod, sim_thresh, window, window + window_size - 1) - desim_reg * sim_below_thresh(cross_prod, sim_thresh, window, window + window_size - 1)
                    j = j + 1
                    
                for window2 in range(int(Z / 2)): # the majority , i.e. the bigger number of uses
                    if j >= len_actual: break
                    window = window + 1    
                    a = sim_above_thresh(cross_prod, sim_thresh, window, window + window_size - 1) - desim_reg * sim_below_thresh(cross_prod, sim_thresh, window, window + window_size - 1)
                    b = sim_above_thresh(cross_prod, sim_thresh, window + 1, window + window_size) - desim_reg * sim_below_thresh(cross_prod, sim_thresh, window + 1, window + window_size)
                    sent_fit[j] = (a + b) / 2
                    window = window + 2
                    j = j + 1
                    
                    
                for window3 in range(quotient + residue):  # the other half of the minority
                    if j >= len_actual: break
                    sent_fit[j] = sim_above_thresh(cross_prod, sim_thresh, window, window + window_size - 1) - desim_reg *  sim_below_thresh(cross_prod, sim_thresh, window, window + window_size - 1)
                    window = window + 1
                    j = j + 1
        
        sent_fit[sent_fit == 0] = np.nan
        batch_fit[i] = np.nanmean(sent_fit) - sent_length_reg * (abs(len_actual - len_desired) / len_desired)
    return np.average(batch_fit)

def sim_desim_fit(sent_length_reg, desim_reg, sim_thresh, ind_thresh, window_size, token2id, embedding_weights):
  def wrapper(y_true, y_pred):
    return sim_desim(y_true, y_pred, sent_length_reg, desim_reg, sim_thresh, ind_thresh, window_size, token2id, embedding_weights)
  return wrapper

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
            word_embedding = np.zeros(embedding_weights.shape[1])
            for k in range(len(word_encoding)):
                try:
                    word_embedding = word_embedding + embedding_weights[token2id[word_encoding[k]]][:]
                except:
                    word_embedding = word_embedding + 0
            if(len(word_encoding) == 0):
                word_embedding = np.zeros(embedding_weights.shape[1])
            else:
                word_embedding = word_embedding / len(word_encoding)
            word_embedding.resize((1,embedding_weights.shape[1]))
            np.put_along_axis(arr=sent_embedding, indices=np.full((1,embedding_weights.shape[1]),j), values=word_embedding, axis=0)
        np.put_along_axis(arr=batch_input_tensor, indices=np.full((1,len(batch_input[0]),embedding_weights.shape[1]),i), values=sent_embedding, axis=0)
    return batch_input_tensor

def calculating_combined_tokens(batch_input, token2id):
    batch_input_tensor = np.zeros((len(batch_input), len(batch_input[0]), len(token2id))) # batch size, sent length, num of tokens
    sent_embedding = np.zeros((len(batch_input[0]), len(token2id)))
    
    for i, sent in enumerate(batch_input):
        sent_embedding.fill(0)
        for j, word in enumerate(sent):
            temp = np.zeros((1,len(token2id)))
            temp[:,token2id[word]] = 1
            np.put_along_axis(arr=sent_embedding, indices=np.full((1,len(token2id)),j), values=temp, axis=0)
        np.put_along_axis(arr=batch_input_tensor, indices=np.full((1,len(batch_input[0]),len(token2id)),i), values=sent_embedding, axis=0)
    return batch_input_tensor

def decode(encoding):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    encoding -- a list containing the tokens 
    
    Returns: 
    decoding -- a string summerizing all tokens.
    '''
    decoding = ''
    for token in encoding.tokens:
        if token == '</w>':
            decoding = decoding + ' '
        else:
            decoding = decoding + token
    return decoding

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

X_Train_enco = encoder_input_data[:cut] + [ ['<PAD>'] * max_enco_input_len for _ in range(batch_size-cut%32)]
X_Test_enco = encoder_input_data[cut:] + [ ['<PAD>'] * max_enco_input_len for _ in range(batch_size-(data_size - cut)%32)]

X_Train_deco = decoder_input_data[:cut] + [ ['<PAD>'] * max_deco_input_len for _ in range(batch_size-cut%32)]
X_Test_deco = decoder_input_data[cut:] + [ ['<PAD>'] * max_deco_input_len for _ in range(batch_size-(data_size - cut)%32)]

Y_Train_deco = decoder_output_data[:cut] + [ ['<PAD>'] * max_deco_output_len for _ in range(batch_size-cut%32)]
Y_Test_deco = decoder_output_data[cut:] + [ ['<PAD>'] * max_deco_output_len for _ in range(batch_size-(data_size - cut)%32)]

# getting the weights of the previous trained model
# TRAINING WITH TEACHER FORCING
encoder_inputs= Input(shape=(max_enco_input_len,embedding_dim), name='encoder_inputs')
encoder_lstm=LSTM(units=50, return_state=True, name='encoder_lstm')
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
model_encoder_training.load_weights('D:\\Project Files\\trained_model_weights.h5')
transfer_learning_model_weights = pygad.kerasga.model_weights_as_vector(model=model_encoder_training)
transfer_learning_model_weights = transfer_learning_model_weights[:140400]

# TRAINING WITH TEACHER FORCING
encoder_inputs= Input(shape=(max_enco_input_len, embedding_dim), name='encoder_inputs')
encoder_lstm=LSTM(units=50, return_state=True, name='encoder_lstm')
LSTM_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# We discard `LSTM_outputs` and only keep the other states.
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_deco_input_len, embedding_dim), name='decoder_inputs')
decoder_lstm = LSTM(units=50, return_sequences=True, return_state=True, name='decoder_lstm')

# Set up the decoder, using `context vector` as initial state.
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

#complete the decoder model by adding a Dense layer with Softmax activation function 
#for prediction of the next output
#Dense layer will output one-hot encoded representation
decoder_dense = Dense(embedding_dim, activation='tanh', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# put together
model_encoder_training = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='model_encoder_training')

model_encoder_training.summary()


# building a pseudo model that represents the last layer which we want to optimize

pseudo_decoder_inputs = Input(shape=(50), name='decoder_inputs')
pseudo_decoder_dense = Dense(embedding_dim, activation='tanh', name='decoder_dense')
pseudo_decoder_outputs = pseudo_decoder_dense(pseudo_decoder_inputs)

pseudo_model = Model(pseudo_decoder_inputs, pseudo_decoder_outputs, name='pseudo_model')

# tf.keras.utils.plot_model(
#     model_encoder_training,
#     to_file="GA_training_model.png",
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=True,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=True
# )

custom_fit = sim_desim_fit(sent_length_reg=0.5, desim_reg=0.2, sim_thresh=0.65, ind_thresh=0.8, window_size=4, token2id=token2id, embedding_weights=embedding_weights)
num_solutions = 50

keras_ga = pygad.kerasga.KerasGA(model=pseudo_model, num_solutions=num_solutions)


data_size = 200 # sentences
X_Train_enco = calculating_combined_embeddings(embedding_weights, X_Train_enco[0:data_size], tokenizer, token2id)
X_Train_deco = calculating_combined_embeddings(embedding_weights, X_Train_deco[0:data_size], tokenizer, token2id)
Y_Train_deco = calculating_combined_embeddings(embedding_weights, Y_Train_deco[0:data_size], tokenizer, token2id)


# As long as the fitness value increases, you can increase the number of generations to achieve better accuracy.
num_generations = 200
num_parents_mating = 15 # Number of solutions to be selected as parents in the mating pool.
initial_population = keras_ga.population_weights # Initial population of network weights.
parent_selection_type = "rws"
crossover_type = "uniform"
crossover_probability = 0.8
mutation_type = "adaptive" # "scramble" # Type of the mutation operator.
mutation_probability = [0.8,0.3]
mutation_percent_genes = 35 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.
# stop_criteria = ["reach_1000000", "saturate_10"]

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       crossover_probability=crossover_probability,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       # stop_criteria=stop_criteria,
                       # parallel_processing=['process',20],
                       on_generation=callback_generation)

# if __name__ == '__main__':
ga_instance.run()

ga_instance.plot_fitness(title="Iteration vs. Fitness", linewidth=4)

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model_encoder_training, weights_vector=solution)
model_encoder_training.set_weights(best_solution_weights)

# calculating training accuracy
y_pred = model_encoder_training.predict_on_batch([X_Train_enco,X_Train_deco])
data_acc = calculating_euclidean_dist(Y_Train_deco, y_pred)
    
data_acc = data_acc / data_size
print("Training Accuracy = ", data_acc)

