from datasets import load_dataset
from transformers import AutoTokenizer

base_model = "gpt2" #
batch_size = 32
vocabulary_size = 2200

def batch_iterator():
    for i in range(0, len(nguni_train), batch_size):
        yield nguni_train[i : i + batch_size]["text"]

#load the dataset
dataset = load_dataset("text", data_dir="/home/jvanschalkwyk/lustre/mlm/data/st")
nguni_train = dataset["train"]

#train the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, mask_token="[MASK]", pad_token="[PAD]")
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=vocabulary_size)

#save the tokenizer
new_tokenizer.save_pretrained("/home/jvanschalkwyk/lustre/mlm/tokenizer/gpt2-st")