
def sample(parameters, token2id, seed, max_output_len): # decoder in inference mode
    """
    Sample a sequence of tokens according to a sequence of probability distributions output of the LSTM decoder

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    token2id -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    # representing the first token which <START> token
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    
    # this is the list which will contain the list of indices of the characters to generate ,i.e. the output
    indices = []
    
    # idx is the index of the one-hot vector x that is set to 1 , all other positions in x are zero
    idx = -1 
    
    # Loop over time-steps t. At each time-step:
    # sample a character from a probability distribution 
    # and append its index (`idx`) to the list "indices". 
    # We'll stop if we reach 50 characters 
    # (which should be very unlikely with a well trained model).
    # Setting the maximum number of characters helps with debugging and prevents infinite loops. 
    time_step = 0
    END_token_id = token2id['<END>']
    
    while (idx != END_token_id and time_step != max_output_len):
        
        # forward propagate x 
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        
        np.random.seed(time_step + seed) 
        
        # sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        indices.append(idx)
        
        # update the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        
        # update "a_prev" to be "a"
        a_prev = a
        
        seed += 1
        time_step +=1

    indices.append((max_output_len -1 -len(indices))*token2id['<PAD>']) # for homogeneity wirh the desierd output and when computing loss
    indices.append(token2id['<END>'])
    
    return indices

def sentences_to_indices(X, token2id, max_len):
    m = X.shape[0]                                   
    X_indices = np.zeros([m,max_len])
    
    for i in range(m):                               
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            if w in token2id:
                X_indices[i, j] = token2id[w]
                j =  j+1
    
    return X_indices

