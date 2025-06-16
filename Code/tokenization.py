from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import re
import json

with open('D:\\Project Files\\data.json') as f:
   data = json.load(f)
   input_data_to_tokenize = [item[:-1] for item in data]
   output_data_to_tokenize = [item[-1] for item in data]
   
with open('D:\\Project Files\\raw_data.txt', 'w') as fp:
    for item in data : 
        for s in item :
            fp.write("%s\n" % s)    # write each item on a new line

tokenizer = Tokenizer(BPE(unk_token = "<UNK>")) #,end_of_word_suffix  = '</w>'
trainer = BpeTrainer(special_tokens = ["<PAD>", "<UNK>", "<SEP>", "<START>", "<END>", '</w>'] , vocab_size = 401)

tokenizer.pre_tokenizer = Whitespace()
# tokenizer.decoder = decoders.BPEDecoder()

tokenizer.train(['D:\\Project Files\\raw_data.txt'], trainer)    

# tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<PAD>"),pad_token ="<PAD>")

tokenizer.save("D:\\Project Files\\tokenizer_trained.json")
tokenizer = Tokenizer.from_file("D:\\Project Files\\tokenizer_trained.json")

token2id = tokenizer.get_vocab()
id2token = {v:k for k, v in token2id.items()}

#creating semi tokenized tokens which is adding the token <SEP> between each sentence and the token </w> between words
semi_tokenized_input_data = list()
temp = ""
i = 0
for item in input_data_to_tokenize :
    for sent in item :
        if i < len(item)-1 :
            temp = temp + sent + "<SEP> "
            i = i + 1
        else :
            temp = temp + sent
    semi_tokenized_input_data.append(temp)
    i = 0 
    temp = ""

temp = list()
for sent in semi_tokenized_input_data:
    temp.append(sent.replace(" ","</w>"))

semi_tokenized_input_data = temp

semi_tokenized_output_data = list()
temp = list()    
for sent in output_data_to_tokenize:
        temp.append(sent.replace(" ","</w>"))

semi_tokenized_output_data = temp

# for i in range(10):
#     print(tokenizer.encode(semi_tokenized_data[randrange(len(semi_tokenized_data))]).tokens)
#     print()

tokenizer.save("D:\\Project Files\\tokenizer_trained.json")

with open("D:\\Project Files\\semi_tokenized_input_data.json", 'w') as f:
    json.dump(semi_tokenized_input_data, f)
    
# deleting whitespace space tag </w> at the end if existant
temp = list()
for sent in semi_tokenized_output_data:
    if sent[-4:-1]+sent[-1] == '</w>':
        temp.append(sent[:-4])
    else:
        temp.append(sent)
semi_tokenized_output_data = temp

with open("D:\\Project Files\\semi_tokenized_output_data.json", 'w') as f:
    json.dump(semi_tokenized_output_data, f)

# flipping the input data for the model
flipped_semi_tokenized_input_data = [(list(reversed(re.split('(</w>)',sent))))[2:] for sent in semi_tokenized_input_data[:27402]]
flipped_semi_tokenized_input_data.extend([(list(re.split('(</w>)',sent)))[2:] for sent in semi_tokenized_input_data[27402:]])
# padding and adding end token <END> to flipped_semi_tokenized_input_data for encoder_input_data
max_len = -56    # 150 + 1
for sent in flipped_semi_tokenized_input_data :
    if max_len < len(sent):
        max_len = len(sent)
        
encoder_input_data = list()
temp = list()
for sent in flipped_semi_tokenized_input_data :
    temp = sent
    temp.extend((max_len - len(sent) - 1)*['<PAD>'])
    temp.extend(['<END>'])
    encoder_input_data.append(temp)
    


# padding and adding start token <START> to semi_tokenized_output_data for decoder_input_data
max_len = -56    # 61 + 1
temp1 = list()
for sent in semi_tokenized_output_data:
    temp1 = [i.tokens for i in tokenizer.encode_batch(re.split('(</w>)',sent))]
    temp2 = list()
    for item in temp1:
        for subitem in item:
            temp2.append(subitem)
    if max_len < len(temp2):
        max_len = len(temp2)

temp1 = list()
temp2 = list()
temp3 = list()
for sent in semi_tokenized_output_data :
    temp1 = [i.tokens for i in tokenizer.encode_batch(re.split('(</w>)',sent))]
    temp2 = list()
    temp2.extend(['<START>'])
    for item in temp1:
        for subitem in item:
            temp2.append(subitem)
    temp2.extend((max_len - len(temp2))*['<PAD>'])
    temp3.append(temp2)
decoder_input_data = temp3

# padding and adding end token <END> to semi_tokenized_output_data for decoder_output_data
max_len = -56    # 61 + 1
temp1 = list()
for sent in semi_tokenized_output_data:
    temp1 = [i.tokens for i in tokenizer.encode_batch(re.split('(</w>)',sent))]
    temp2 = list()
    for item in temp1:
        for subitem in item:
            temp2.append(subitem)
    if max_len < len(temp2):
        max_len = len(temp2)

temp1 = list()
temp2 = list()
temp3 = list()
for sent in semi_tokenized_output_data :
    temp1 = [i.tokens for i in tokenizer.encode_batch(re.split('(</w>)',sent))]
    temp2 = list()
    for item in temp1:
        for subitem in item:
            temp2.append(subitem)
    temp2.extend((max_len - len(temp2) - 1)*['<PAD>'])
    temp2.extend(['<END>'])
    temp3.append(temp2)
decoder_output_data = temp3

with open("D:\\Project Files\\encoder_input_data.json", 'w') as f:
    json.dump(encoder_input_data, f)
    
with open("D:\\Project Files\\decoder_input_data.json", 'w') as f:
    json.dump(decoder_input_data, f)

with open("D:\\Project Files\\decoder_output_data.json", 'w') as f:
    json.dump(decoder_output_data, f)

# doc_data = list()
# temp = ""
# i = 0
# for item in data :
#     for sent in item :
#             temp = temp + sent
#     doc_data.append(temp)
#     i = 0 
#     temp = ""
 
# with open("D:\\Project Files\\doc_data.json", 'w') as f:
#     json.dump(doc_data, f)
    
    

