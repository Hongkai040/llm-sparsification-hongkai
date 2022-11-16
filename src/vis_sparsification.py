import transformers 
from transformers import GPT2Model, GPT2Tokenizer 
from transformers import BartModel, BartTokenizer 
from transformers import BertModel, BertTokenizer 

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.utils.prune as prune

import copy
import json
import os
from itertools import chain
import time


def get_params_plot(model, model_name, sparsity=0):
  # get params dist.
  fig, ax = plt.subplots(figsize=(12,8))
  total_param = torch.cat([params.flatten() for params in model.parameters()]).detach().numpy()
  hist, bins = np.histogram(total_param, bins = np.linspace(-1,1,201)) 
  ax.plot(bins[:-1],hist) 
  plt.xlabel("Weight values")
  plt.ylabel("Log Count")
  plt.yscale('log')
  plt.title("distribution of all weights(model:{}, sparsity:{}%)".format(model_name, sparsity*100), fontsize=25)
  plt.savefig('{}/figs/vis_all_params_{}_sparsity_{}%.png'.format(os.getcwd(), model_name, sparsity*100))
  
  # get params dist. in each layer
  for elem in model.named_children():
    if elem[0] in ['h', 'encoder', 'decoder']: 
      fig, ax = plt.subplots(figsize=(12,8))
      for idx, child in enumerate(elem[1].named_children()):
        layer_param = torch.cat([params.flatten() for params in child[1].parameters()]).detach().numpy()
        hist, bins = np.histogram(layer_param, bins = np.linspace(-1,1,201)) 
        ax.plot(bins[:-1],hist,label="{} hidden layer".format(idx)) 
      plt.xlabel("Weight values")
      plt.ylabel("Log Count")
      plt.yscale('log')
      plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
      plt.title("distribution of weights across layers(model:{}, sparsity:{}%)".format(model_name, sparsity*100), fontsize=25)
      plt.savefig('{}/figs/vis_layers_params_{}_sparsity_{}%.png'.format(os.getcwd(), model_name, sparsity*100))


def prune_model(model, model_name, prune_portion):
  '''
  using global pruning to prune gpt2, bart, and bert. 
  '''
  def print_sparsity(module_lst, prune_flag):

    zero_params_count = sum([torch.sum(module[0].weight == 0) for module in module_lst])
    total_params_count = sum([module[0].weight.nelement() for module in module_lst])

    print(
        "Global sparsity of model: {} in hidden layers {} pruning: {:.2f}%".format(model_name, prune_flag,
                                                    100. * float(zero_params_count)
                                                        / float(total_params_count)
                                                    )
        )
    return None
    
  if model_name == 'BART-large':
    modules = chain(model.encoder.modules(), model.decoder.modules())
  elif model_name == 'BERT-large':
    modules = model.encoder.modules()
  else:
    modules = model.h.modules()

  module_lst = []
  for module in modules:
    if hasattr(module, 'weight'):
      module_lst.append((module, 'weight'))

  print_sparsity(module_lst, prune_flag = 'before')
  prune.global_unstructured(module_lst, 
                            pruning_method=prune.L1Unstructured, 
                            amount=prune_portion)
  for module, attr in module_lst: 
    prune.remove(module, attr)
  print_sparsity(module_lst, prune_flag = 'after')
  return model

# Method from https://discuss.pytorch.org/t/finding-model-size/130275/5
def get_model_size(model):
  param_size = 0
  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))
  return size_all_mb

if __name__ == "__main__":
  model_names = ['GPT2-medium', 'BART-large', 'BERT-large']
  sparsity = [0, 0.1, 0.5, 0.9, 0.95, 0.99] 
  # model_names = ['BERT-large']
  # sparsity = [0.5] 
  model_size_dic = dict()

  t0 = time.time()
  print("Start sparsifying...")
  for model_name in model_names:
    if model_name == 'GPT2-medium':
      model = GPT2Model.from_pretrained('gpt2-medium')
      # model = GPT2Model.from_pretrained('gpt2')
    elif model_name == 'BART-large':
      model = BartModel.from_pretrained("facebook/bart-large")
      # model = BartModel.from_pretrained("facebook/bart-base")
    else:
      # model = BertModel.from_pretrained("bert-base-cased")
      model = BertModel.from_pretrained("bert-large-cased")

    for prune_proportion in sparsity:
      target_model = copy.deepcopy(model)
      if prune_proportion != 0:
        target_model = prune_model(target_model, model_name, prune_proportion)
      model_size_dic[f'{model_name}_sparsity_{prune_proportion*100}%'] = get_model_size(target_model)
      get_params_plot(target_model, model_name, prune_proportion)
      target_model.save_pretrained("/home/hongkai/DLS/results/models/{}_sparsity_{}".format(
                                                              model_name,
                                                              prune_proportion*100,
                                                              ))
      print("model {} with sparsity {} saved".format(model_name, prune_proportion * 100))
      print('{0:.3f} minutes elapsed'.format((time.time() - t0)/60))
      
  print("Sparsification completed!")

  dir_open = open(f'{os.getcwd()}/model_size.json', "w")
  json.dump(model_size_dic, dir_open, indent = 4)