import transformers
import wandb
import torch
import math
from evaluate import load
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig, GPT2LMHeadModel
from datasets import load_dataset, ClassLabel, load_metric

#language dataset
language = "zu"

#login to wandb for logging
wandb.login()

#configurations for model
hidden_size=768
hidden_dropout_prob=0.1
num_hidden_layers=8
num_att_heads=8
vocab_size=8000

#configurations for the hyperparameter optimization sweep
config = {
    'method':'bayes',
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
    
    'd_prob': {
        'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
    
    'hidden_size': {
        'values': [144, 288, 384, 512, 768]
    },
    
    'warmup_steps': {
        'values': [500, 1000, 1500, 2000, 2500, 3000]
    }
    
}

config['parameters'] = parameters
sweep_id = wandb.sweep(config, project='huggingface')

#main checkpoints
datasets = load_dataset("text", data_dir=f"data/{language}")
model_checkpoint = "gpt2"
tokenizer_checkpoint = "tokenizer/gpt2"

#loading all important informarion
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
block_size = 128

def tokenize_function(examples):
    return tokenizer(examples["text"])#stride=16
 
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
def train(config=None):
    with wandb.init(config=config):

        config=wandb.config    
        tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
   
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=config.batch_size,
            num_proc=4,
        )   
    
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model_config = AutoConfig.from_pretrained(
            model_checkpoint, 
            vocab_size=vocab_size,
            n_embd=config.hidden_size,
            n_layer=num_hidden_layers,
            n_head=num_att_heads,
            embd_pdrop=config.d_prob
            )
        model = GPT2LMHeadModel(config)
        model = model.to(device)
    
        training_args = TrainingArguments(
            output_dir="sweep_gpt",
            report_to='wandb',
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            optim="adamw_torch",
            max_steps=20000,
            warmup_steps=config.warmup_steps,
            learning_rate=1e-4,
            weight_decay=0.3,
            eval_steps=5000,
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
