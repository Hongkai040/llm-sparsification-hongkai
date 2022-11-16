import transformers 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM 
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

import torch
import numpy as np 

import datasets
from datasets import load_metric, load_dataset

import json
import os 
import time 

START_TIME = time.time()

def run_benchmarks(models, sparsity, tokenizers, model_names):
  '''
  run benchmarks
    GLUE(qnli) and CLM(wikitext v2 raw)
  '''
  # from https://ifwind.github.io/2021/08/31/BERT实战——（7）生成任务-语言模型/#调用分词器对所有的文本分词
  # for  CLM
  def group_texts(examples):
    block_size = 256
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
  
  # CLM
  def tokenize_function_clm(examples):
    return tokenizer(examples["text"], truncation=True)

  # GLUE
  def tokenize_function_glue(examples):
    return tokenizer(examples["question"], 
                     examples["sentence"],
                     padding="max_length",
                     max_length=256, 
                     truncation=True)
  
  # for GLUE 
  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

  # eval_pred of bart is different
  def compute_metrics_bart(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)
    return metric.compute(predictions=predictions, references=labels)

  for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(tokenizers[model_name])
    if model_name == 'GPT2-medium':
      tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    for idx, model in enumerate(models[model_name]):
      for task in ['clm', 'glue']:
        if task == 'clm':
          dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')
          tokenized_datasets = dataset.map(tokenize_function_clm, batched=True, remove_columns=["text"])
          tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000)
          model = AutoModelForCausalLM.from_pretrained(models[model_name][idx])
        else:
          dataset = datasets.load_dataset('glue', 'qnli')
          metric = load_metric("glue", "qnli")
          tokenized_datasets = dataset.map(tokenize_function_glue, batched=True)
          # change number of labels accordingly
          model = AutoModelForSequenceClassification.from_pretrained(models[model_name][idx], 
                                                                    num_labels=2)
        if model_name == 'GPT2-medium':
          model.config.pad_token_id = model.config.eos_token_id

        train_dataset = tokenized_datasets["train"].shuffle(seed= 42).select(range(5000))
        eval_dataset = tokenized_datasets["validation"].shuffle(seed= 42).select(range(800))
        training_args = TrainingArguments(output_dir=f"{os.getcwd()}/training/{model_name}_{task}_sparsity_{sparsity[idx]*100}_output", 
                                    num_train_epochs=1,
                                    per_device_train_batch_size=32,
                                    per_device_eval_batch_size=32,
                                    save_strategy = "steps",
                                    save_steps = 10000,
                                    evaluation_strategy="epoch")
        
        if task == 'clm':
          trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            data_collator=data_collator,
                            )
        else:
          if model_name == 'BART-large':
            trainer = Trainer(
                              model=model,
                              args=training_args,
                              train_dataset=train_dataset,
                              eval_dataset=eval_dataset,
                              data_collator=data_collator,
                              compute_metrics=compute_metrics_bart,
                              )
          else:
            trainer = Trainer(
                              model=model,
                              args=training_args,
                              train_dataset=train_dataset,
                              eval_dataset=eval_dataset,
                              data_collator=data_collator,
                              compute_metrics=compute_metrics,
                              )

        train_results = trainer.train()
        eval_results = trainer.evaluate()
        result = {'train': train_results.metrics,
                  'eval': eval_results}
        dir_open = open(f'{os.getcwd()}/benchmark/{model_name}_{task}_sparsity_{sparsity[idx]*100}_results.json', "w")
        json.dump(result, dir_open, indent = 4)
        print('{0:.3f} minutes elapsed'.format((time.time() - START_TIME)/60))


if __name__ == "__main__":
    model_names = ['GPT2-medium', 'BART-large', 'BERT-large']
    tokenizers_address = ['gpt2-medium', 'facebook/bart-large', "bert-large-cased"]
    # sparsity = [0, 0.1, 0.5, 0.9, 0.95, 0.99] 
    sparsity = [0.5] 

    models = dict()
    tokenizers = dict()
    for model_name, tokenizer_add in zip(model_names, tokenizers_address):
        tokenizers[model_name] = tokenizer_add
        models[model_name] = []
        for prune_proportion in sparsity:
            models[model_name].append(f'/home/hongkai/DLS/results/models/{model_name}_sparsity_{prune_proportion*100}')

    run_benchmarks(models, sparsity, tokenizers, model_names)