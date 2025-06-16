import multiprocessing

import numpy as np

from tokenizers import Tokenizer

from sklearn.decomposition import PCA

from matplotlib import pyplot
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from random import randrange
import json

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

with open('D:\\Project Files\\raw_data.txt') as f:
    raw_data = f.readlines()
    
with open('D:\\Project Files\\semi_tokenized_input_data_4_embeddings.json') as f:
    data1 = json.load(f)
    
with open('D:\\Project Files\\semi_tokenized_output_data_4_embeddings.json') as f:
    data2 = json.load(f)
    
tokenizer = Tokenizer.from_file("D:\\Project Files\\tokenizer_trained.json")

token2id = tokenizer.get_vocab()
id2token = {v:k for k, v in token2id.items()}

sentences = list()
for i in range(len(data1)):
        sentences.append(data1[i])
        sentences.append(data2[i])
        
w2v = Word2Vec(sentences, vector_size = 300, sg=1, negative=15, epochs=100, window=4, alpha=0.05, min_count=0,
               workers=multiprocessing.cpu_count(), compute_loss=True, callbacks=[callback()])

weights = w2v.syn1neg  # missing tokens in the embedding weights ['abo', 'peop', 'yel']

# for i in range(6):
#     x = np.random.normal(0, 0.005, (300))
#     weights = np.insert(weights,0,[x],axis= 0)
    
np.save("D:\\Project Files\\embedding_weights.npy", weights)

# calculating combined weights for each word and unique words which is equal to 6420
unique_words = set()
for sent in raw_data:
    for word in sent.split():
        unique_words.add(word)

weights = np.delete(weights, [0,1,2,3,4,5], axis=0)
temp = list()
w = 0
combined_weights = np.empty((len(unique_words),weights.shape[1]))

j = 0
for word in unique_words:
    temp = tokenizer.encode(word).tokens
    w = 0
    for i in range(len(temp)):
        try:
            w = w + weights[token2id[temp[i]]][:]
        except:
            w = w + 0  
    combined_weights[j][:] = w / len(temp)
    j = j + 1

pca = PCA(n_components=2)
result = pca.fit_transform(combined_weights)

rand_choice_words = list()
for i in range(10):
    rand_choice_words.append(randrange(len(unique_words)))

for i in rand_choice_words:
    pyplot.scatter(result[i, 0], result[i, 1])
    
words = list(unique_words)

for word in rand_choice_words:
	pyplot.annotate(words[word], xy=(result[word, 0], result[word, 1]))
pyplot.show()