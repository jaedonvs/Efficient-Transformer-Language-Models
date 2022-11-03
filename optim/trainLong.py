import transformers
import wandb
import torch
import math
import functools
from evaluate import load
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig
from datasets import load_dataset, ClassLabel, load_metric

#language dataset
language = "zu"

#login to wandb for logging
wandb.login()

#configurations for model
hidden_size=768
hidden_dropout_prob=0.1
num_hidden_layers=8
num_attention_heads=8
vocab_size=8000

#configurations for the hyperparameter optimization sweep
config = {
    'method':'bayes', #change to bayes
    }

#metric
metric = {
    'name': 'eval/loss',
    'goal': 'minimize'
}
config['metric'] = metric

#parameters
parameters = {
    'batch_size': { 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 512,
      },
    
    'big_block_size': { 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 128,
        'max': 1024,
      },

    'hidden_size': { #multiples of attention head
        'values': [512, 768]
        },
    
    'attention_window':{
        'values': [8, 16, 24, 32, 48, 56]
        }
    }

config['parameters'] = parameters
sweep_id = wandb.sweep(config, project='huggingface')

#main checkpoints
datasets = load_dataset("text", data_dir=f"data/{language}")
model_checkpoint = "allenai/longformer-base-4096"
tokenizer_checkpoint = "tokenizer/longformer"

#loading all important informarion
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
block_size = 768 #

def tokenize_function(examples):
    return tokenizer(examples["text"])#stride=16
 
def group_texts(examples, block_size):
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
def train(config=None):
    with wandb.init(config=config):

        config=wandb.config    
        tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
   
        lm_datasets = tokenized_datasets.map(
            functools.partial(group_texts, block_size = config.big_block_size),
            batched=True,
            batch_size=config.batch_size,
            num_proc=4,
        )   
    
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model_config = AutoConfig.from_pretrained(
            model_checkpoint, 
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            
            hidden_size=hidden_size,
            attention_window=attention_window#
            )
        model = AutoModelForMaskedLM.from_config(model_config)
        model = model.to(device)
    
        training_args = TrainingArguments(
            output_dir="sweep_long",
            report_to='wandb',
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            optim="adamw_torch",
            eval_steps=5000,
            max_steps=20000,
            learning_rate=1e-4,
            warmup_steps=1500,
            weight_decay=0.3,
            fp16=True,
        )
    
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
            data_collator=data_collator,
        )
    
        print("Starting Training...")
        result = trainer.train()

        print("Training results:")
        print(result)

        print("Evaluation results...")
        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        trainer.save_model('/home/jvanschalkwyk/lustre/optim/savedModel')
    
if __name__ == "__main__":
    wandb.agent(sweep_id, train, count=20)   
