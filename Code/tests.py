from tokenizers import Tokenizer
import numpy
from numpy import dot
from numpy.linalg import norm
import json

tokenizer = Tokenizer.from_file("D:\\Project Files\\tokenizer_trained.json")

token2id = tokenizer.get_vocab()
id2token = {v:k for k, v in token2id.items()}

weights = numpy.load("D:\\Project Files\\embedding_weights.npy")

with open('D:\\Project Files\\doc_data.json') as f:
   doc_data = json.load(f)

sentences = [[token for token in tokenizer.encode(rel).tokens] for rel in doc_data]


y = tokenizer.encode('people').tokens
x = 0
for i in range(len(y)):
    x = x + weights[token2id[y[i]]][:]
x = x / len(y)

y = tokenizer.encode('laptop').tokens  #yellow
z = 0
for i in range(len(y)):
    z = z + weights[token2id[y[i]]][:]
z = z / len(y)

y = tokenizer.encode('screen').tokens  # about
m = 0
for i in range(len(y)):
    m = m + weights[token2id[y[i]]][:]
m = m / len(y)

result = dot(z, m)/ (norm(z)*norm(x))

def decode(encoding):
    decoding = ''
    for token in encoding.tokens:
        if token == '</w>':
            decoding = decoding + ' '
        else:
            decoding = decoding + token
    return decoding