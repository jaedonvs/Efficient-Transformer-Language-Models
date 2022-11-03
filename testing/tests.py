import math
import torch
import collections
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, ClassLabel
from evaluate import evaluator, load
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForMaskedLM,
    AutoModel,
    DataCollatorForLanguageModeling,
    pipeline,
    default_data_collator,
    TrainingArguments,
    Trainer
)

model_version = '/home/jvanschalkwyk/lustre/mlm/saved/gpt2_final_zu'
tokenizer_version = '/home/jvanschalkwyk/lustre/mlm/tokenizer/gpt2'
data_dir = '/home/jvanschalkwyk/lustre/testBPE/data/zu'

model = AutoModel.from_pretrained(model_version).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_version)
datasets = load_dataset("text", data_dir=data_dir)
mask_filler = pipeline('fill-mask', model=model_version, tokenizer=tokenizer)

num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> Number of parameters: {round(num_parameters)}M'")


# text = 'Sithole uhlu lwemindeni ephila [MASK] yobumpofu'
# inputs = tokenizer(text, return_tensors="pt")
# token_logits = model(**inputs).logits
# # Find the location of [MASK] and extract its logits
# mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
# mask_token_logits = token_logits[0, mask_token_index, :]
# # Pick the [MASK] candidates with the highest logits
# top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
# for token in top_5_tokens:
#     print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

# test_data = datasets["test"]
# print(test_data)
# counter = 0

# for i in range(7301):
#     print(f'Run {i}')
#     tester = tokenizer(test_data['text'][i])
#     counter += len(tester['input_ids'])
    
# print(f"total number of tokens: {counter}")

# tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])

# chunk_size = 128
# def group_texts(examples):
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     total_length = (total_length // chunk_size) * chunk_size
#     result = {
#         k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

# lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# batch_size = 64
# training_args = TrainingArguments(
#     output_dir="/home/jvanschalkwyk/lustre/testBPE/junk",
#     overwrite_output_dir=True,
#     evaluation_strategy="steps",
#     max_steps=50,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=lm_datasets["train"],
#     eval_dataset=lm_datasets["test"],
#     data_collator=data_collator,
# )

#eval_results = trainer.evaluate()
#print(eval_results)

#print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")




# # #Sithole uhlu lwemindeni ephila impilo yobumpofu
text = 'Sithole uhlu lwemindeni ephila [MASK] yobumpofu'


preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred}")






# test = load_dataset("text", data_dir=data_dir, split="test")
# encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")


# max_length = 512
# stride = 32
# seq_len = encodings.input_ids.size(1)
# device = "cuda:0"

# nlls = []
# prev_end_loc = 0
# for begin_loc in tqdm(range(0, seq_len, stride)):
#     end_loc = min(begin_loc + max_length, seq_len)
#     trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
#     input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#     target_ids = input_ids.clone()
#     target_ids[:, :-trg_len] = -100

#     with torch.no_grad():
#         outputs = model(input_ids, labels=target_ids)

#         # loss is calculated using CrossEntropyLoss which averages over input tokens.
#         # Multiply it with trg_len to get the summation instead of average.
#         # We will take average over all the tokens to get the true average
#         # in the last step of this example.
#         neg_log_likelihood = outputs.loss * trg_len

#     nlls.append(neg_log_likelihood)

#     prev_end_loc = end_loc
#     if end_loc == seq_len:
#         break

# ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
# print(ppl)