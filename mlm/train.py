import argparse
import math
import wandb
import torch
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForMaskedLM,
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

#----------------------------------args----------------------------------#
parser = argparse.ArgumentParser(description="Training a language model")
parser.add_argument("--data_dir", type=str, default="data/zu", help="Path to the data directory")
parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Model checkpoint to use")
parser.add_argument("--tokenizer_checkpoint", type=str, default="tokenizer/gpt2", help="Tokenizer checkpoint to use")
parser.add_argument("--block_size", type=int, default=64, help="Block size")
parser.add_argument("--output_dir", type=str, default="output_checkpoints", help="Output directory")
parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy")
parser.add_argument("--max_steps", type=int, default=50000, help="Max steps for evaluation")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="Gradient checkpointing")
parser.add_argument("--fp16", type=bool, default=False, help="Use fp16")
parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation Steps")
parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
args=parser.parse_args()

#initialize w&b
project_text = args.tokenizer_checkpoint
wandb.init(project="gpt-2", entity="mlm", id="ZuFalse")

#Model Configurations
hidden_size=768
hidden_dropout_prob=0.1
num_hidden_layers=8
num_att_heads=8
vocab_size=8000 #8000 for zu and 2200 for st

#main checkpoints
datasets = load_dataset("text", data_dir=args.data_dir)
model_checkpoint = args.model_checkpoint
tokenizer_checkpoint = args.tokenizer_checkpoint

#loading all important informarion
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
block_size = args.block_size

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, stride=16)
 
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
  
#data processing and training
def MLM_training():
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=32,
        num_proc=4,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"   
    config = AutoConfig.from_pretrained(
        model_checkpoint,
        vocab_size = vocab_size,
        n_embd = hidden_size,
        n_layer = num_hidden_layers,
        n_head = num_att_heads,
        embd_pdrop = hidden_dropout_prob
        )
    model = GPT2LMHeadModel(config) #AutoModelForMaskedLM
    model = model.to(device)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy = args.evaluation_strategy,
        max_steps = args.max_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        save_steps=args.save_steps,
        optim="adamw_torch",
        report_to="wandb",
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )
    
    trainer.train()
    evaluation(trainer)
    trainer.save_model('/home/jvanschalkwyk/lustre/mlm/saved/gpt2_final_zu_false')
    
def evaluation(trainer):
    print("Starting evaluation")
    eval_results = trainer.evaluate()
    loss = eval_results['eval_loss']
    print("| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc TBC!!! {:8.2f}".format(
        loss,
        math.exp(loss),
        loss / math.log(2) / 4
    ))
   
if __name__ == "__main__":
    MLM_training()