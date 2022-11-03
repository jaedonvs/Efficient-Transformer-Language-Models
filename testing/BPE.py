import data

dataset = "data/st"
vocab_size = 2200

def get_avg_char_per_token():
    corpus = data.Corpus(dataset, vocab_size, True)
    return corpus.dictionary.avg_characters_per_token.get('test')

if __name__=="__main__":
    print(get_avg_char_per_token())
