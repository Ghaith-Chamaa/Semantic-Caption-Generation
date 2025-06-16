from tokenizers import Tokenizer
import numpy as np
from math import floor, ceil

def index_of_vec(vec, arr):
    ind=0
    for ind,vec2 in enumerate(arr):
        temp = vec == vec2
        if np.all(temp == True):
            return ind
    return 1e3

def minimum(candidates, beg, end):
    mini = 1e9
    i = beg
    while i <= end and i != len(candidates):
        if candidates[i] < mini:
            mini = candidates[i]
        i+=1
    return mini

def maximum(candidates, beg, end):
    maxi = -1e9
    i = beg
    while i <= end and i != len(candidates):
        if candidates[i] > maxi:
            maxi = candidates[i]
        i+=1
    return maxi

def vector_matrix_cos_sim(a, B):
    "" "cosine similarity between vector and matrix" ""
    return np.dot(B, a) / (np.linalg.norm(a) * np.linalg.norm(B, axis=1))

def sim_desim(y_true, y_pred, sent_length_reg, sim_reg, window_size, token2id, embedding_weights):
    # y_true = tf.make_tensor_proto(y_true)
    # y_true = tf.make_ndarray(y_true)
    # y_pred = tf.make_tensor_proto(y_pred)
    # y_pred = tf.make_ndarray(y_pred)
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    
    batch_cost = np.zerors(y_true.shape[0]) # batch size
    y_pred_embedded = np.ndarray(y_true.shape) # batch size, sent length, word embedding dim
    
    # transforming y_pred into a tensor of same shape as y_true
    pad_embedding = embedding_weights[token2id['<PAD>']][:]
    end_embedding = embedding_weights[token2id['<END>']][:]
    pad_embedding.resize((1,300))
    end_embedding.resize((1,300))
    sent_embedding = np.zeros(y_true.shape[1:]) # sent length, word embedding dim
    word_embedding = np.zeros((1,y_true.shape[2]))
    for i, sent in enumerate(y_pred): # for each sent in the batch (first dim)
        sent_embedding.fill(0)
        token_ids = np.argmax(sent, axis=1)
        for j, token_id in enumerate(token_ids):
            word_embedding = embedding_weights[token_id,:]
            word_embedding.resize((1,embedding_weights.shape[1]))
            np.put_along_axis(arr=sent_embedding, indices=np.full((1,embedding_weights.shape[1]),j), values=word_embedding, axis=0)
        np.put_along_axis(arr=y_pred_embedded, indices=np.full((1,y_pred.shape[1],embedding_weights.shape[1]),i), values=sent_embedding, axis=0)
    
    for i, sent in enumerate(y_pred_embedded): # for each sent in the batch
        len_actual = min(index_of_vec(pad_embedding, y_pred_embedded[i,:,:]), index_of_vec(end_embedding, y_pred_embedded[i,:,:]), y_pred_embedded.shape[1])
        len_desired = min(index_of_vec(pad_embedding, y_true[i,:,:]), index_of_vec(end_embedding, y_true[i,:,:]), y_true.shape[1])
        sent_cost = np.zeros(len_actual)
        if len_actual >= len_desired:
            X = len_desired - window_size + 1 # num of windows available in desired
            Y = len_actual / X # num of window uses
            Z = len_actual % X # residue
            # there are Z windows that will be used ceil(Y) times , and
            # X - Z windows that will be used floor(Y) times
            for j, word in enumerate(sent): # for each word in the sent
                if j == len_actual - 1:
                    break
                cross_prod = vector_matrix_cos_sim(y_pred_embedded[i,j,:],(y_true)[i,:,:]) # pred word embedding X all embeddings of the words of the true sent
                # cross_prod.resize(1,cross_prod.shape[0])
                window = 0
                quotient , residue = divmod(X - Z, 2)
                for window in range(quotient):  # half of the minority , i.e. the lower number of uses
                    for usage in range(floor(Y)):
                        sent_cost[j] = np.nanmin(cross_prod[window : window + window_size - 1]) - sim_reg * np.nanmax(cross_prod[window : window + window_size - 1])
                                       
                for window2 in range(Z): # the majority , i.e. the bigger number of uses
                    window = window + window2
                    for usage in range(ceil(Y)): 
                        sent_cost[j] = np.nanmin(cross_prod[window : window + window_size - 1]) - sim_reg * np.nanmax(cross_prod[window : window + window_size - 1])
                
                for window3 in range(quotient + residue):  # the other half of the minority
                    window = window + window3
                    for usage in range(floor(Y)):
                        sent_cost[j] = np.nanmin(cross_prod[window : window + window_size - 1]) - sim_reg *  np.nanmax(cross_prod[window : window + window_size - 1])
        else:
            X = len_desired - window_size + 1 # num of windows available in desired
            Y = len_actual / X # num of window uses
            Z = (X - len_actual) * 2 # residue
            # there are X - Z windows that will be used once (always) , and
            # Z windows that will be used Z / 2 times
            for j, word in enumerate(sent): # for each word in the sent
                if j == len_actual - 1:
                    break
                cross_prod = vector_matrix_cos_sim(y_pred_embedded[i,j,:],(y_true)[i,:,:]) # pred word embedding X all embeddings of the words of the true sent
                # cross_prod.resize(1,cross_prod.shape[0])
                window = 0
                quotient , residue = divmod(X - Z, 2)
                for window in range(quotient):  # half of the minority , i.e. the lower number of uses
                    sent_cost[j] = minimum(cross_prod, window, window + window_size - 1) - sim_reg *  np.nanmax(cross_prod[window : window + window_size - 1])
                                       
                for window2 in range(int(Z / 2)): # the majority , i.e. the bigger number of uses
                    a = np.nanmin(cross_prod[window : window + window_size - 1]) - sim_reg *  np.nanmax(cross_prod[window : window + window_size - 1])
                    b = np.nanmin(cross_prod[window + 1 : window + window_size]) - sim_reg *  np.nanmax(cross_prod[window + 1 : window + window_size])
                    sent_cost[j] = (a + b) / 2
                    window = window + 1
                    
                for window3 in range(quotient + residue):  # the other half of the minority
                    window = window + window3
                    sent_cost[j] = np.nanmin(cross_prod[window : window + window_size - 1]) - sim_reg *  np.nanmax(cross_prod[window : window + window_size - 1])
        
        batch_cost[i] = np.average(sent_cost) - sent_length_reg * abs(len_actual - len_desired)
    return batch_cost

def sim_desim_loss(sent_length_reg, sim_reg, window_size, token2id, embedding_weights):
  def wrapper(y_true, y_pred):
    return -sim_desim(y_true, y_pred, sent_length_reg, sim_reg, window_size, token2id, embedding_weights)
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