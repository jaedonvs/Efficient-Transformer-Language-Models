from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel, BigBirdModel, BigBirdTokenizer, LongformerModel, LongformerTokenizer, AutoTokenizer, GPT2LMHeadModel, BigBirdForCausalLM, FlaxBigBirdModel, FlaxBigBirdForCausalLM, BigBirdTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt

zulu = "Loku kuzakuqinisekisa ukuthi lohlelo luzakubanathi isikhathi eside kanti futhi uhulumeni uzakukwazi ukuthi angeze imali yokusimamisa lezinhlelo ukuze kudaleke amanye amathuba emisebenzi namathuba okuvula amabhizinisi. Siyaziqhenya ngukuthi esikhathini esingangonyaka sikwazile ukunciphisa ngesigamu umsebenzi obungakaqedwa wokuthutha indle ngamabhakede ezindaweni lapho kuhlala khona abantu . Sizakuqinisekisa ukuthi izinqumo zokukhulisa amajele , nokwelusa imincele yethu , namaphepha okuhamba , konke loku kuzakufezekiswa"

sotho = "Heke ya tshireletso ka hara ntlo, e arolang diphaposi tsa ho robala le diphaposi tse ding tsa ntlo, e ka ba bohlokwa haholo ha ho ka etseha hore ho be le motho ya ka kenang ka tlung. Ho tshwanetswe hore ho kengwe le sesebediswa sa ho hokahana le batho ba bang tlasa maemo a tshohanyetso sebakeng sena. Ka dinako tsohle, dithunya di tshwanela ho ba dibakeng tseo di ka fumanehang ha bonolo ha di batleha, mme ha ho kgoneha se ka dula mothong. Etsang bonnete ba hore ka mehla le isa dibetsa tsa lona ho ya hlahlojwa hore di ntse di sebetsa hantle"

#----------------------------------------model versions------------------------------------#
# model_version = '/home/jvanschalkwyk/lustre/testBPE/saved/bert_final_zu'
# model_version="gpt2"
# tokenizer_version = '/home/jvanschalkwyk/lustre/testBPE/tokenizer/gpt2'
# sentence = "Ukusa kwaziwa yibona sebephi kelele, Ukusa kwaziwa yibona sebephi kelele, Ukusa kwaziwa"

# model = GPT2LMHeadModel.from_pretrained(model_version, output_attentions=True)
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_version)
# inputs = tokenizer.encode(sentence, return_tensors='pt')
# input_ids = inputs
# attention = model(input_ids)[-1]
# input_id_list = input_ids[0].tolist() # Batch index 0
# tokens = tokenizer.convert_ids_to_tokens(input_id_list) 


model_version = '/home/jvanschalkwyk/lustre/mlm/saved/bigbird_small_zu'
tokenizer_version = '/home/jvanschalkwyk/lustre/testBPE/tokenizer/bigbird'
sentence = "Ukusa kwaziwa yibona sebephi kelele, Ukusa kwaziwa yibona sebephi kelele, Ukusa kwaziwa, Ukusa kwaziwa, Ukusa kwaziwa, Ukusa kwaziwa, Ukusa"

model_version='google/bigbird-roberta-base'
model = FlaxBigBirdForCausalLM.from_pretrained(model_version, output_attentions=True)
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
#model = BigBirdModel.from_pretrained(model_version, output_attentions=True)
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_version)
inputs = tokenizer.encode(sentence, return_tensors='pt')
input_ids = inputs
out = model(input_ids)
print(out)
#attention = model(input_ids)[-1]
#print(attention)

input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list) 

# model_version = '/home/jvanschalkwyk/lustre/mlm/saved/longformer_small'
# tokenizer_version = '/home/jvanschalkwyk/lustre/testBPE/tokenizer/longformer'
# sentence = "Ukusa kwaziwa yibona sebephi kelele, Ukusa kwaziwa yibona sebephi kelele, Ukusa kwaziwa"

# model = LongformerModel.from_pretrained(model_version, output_attentions=True)
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_version)
# inputs = tokenizer.encode(sentence, return_tensors='pt')
# input_ids = inputs
# attention = model(input_ids)[-1]
# input_id_list = input_ids[0].tolist() # Batch index 0
# tokens = tokenizer.convert_ids_to_tokens(input_id_list) 

#------------------------------------------------------------------------------#

#For masking
# for k in range(8):
# 	for i in range(27):
# 		for j in range(i+1, 27):
# 			attention[0][0][k][i][j] = 0

# for i in range(8):
# 	counter=0
# 	for j in range(24, -1, -1):
# 		for k in range(12):
# 			attention[0][0][i][counter][j+k] = 0
# 		counter+=1
        
 

head = 0
attention_heads = torch.squeeze(attention[0], 0)
attention2 = attention_heads[0]

def plot_attention_head(in_tokens, translated_tokens, attention):
	translated_tokens = translated_tokens[1:]

	ax = plt.gca()
	with torch.no_grad():
		ax.matshow(attention)
	ax.set_xticks(range(len(in_tokens))) #swapped x and y
	ax.set_yticks(range(len(translated_tokens)))

	labels = [label for label in in_tokens]
	ax.set_xticklabels([], rotation=90)

	labels = [label for label in translated_tokens]
	ax.set_yticklabels([])



def plot_attention_weights(tokens, att):
	fig = plt.figure(figsize=(32, 16))

	for h in range(7):
		ax = fig.add_subplot(3, 4, h+1)
		attention_heads = torch.squeeze(att[0], 0)
		attention = attention_heads[h]
		plot_attention_head(tokens, tokens, attention)

	plt.tight_layout()
	#plt.show()
	fig.savefig('graphs/flaxbb.png')


#plot_attention_head(tokens, tokens, attention2)
plot_attention_weights(tokens, attention)