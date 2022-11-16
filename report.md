# Lab4 LLM Sparsification

## Concepts

* Sparsification: Making the network sparser by assigning zero weights. The model size may not be reduced. 
* Pruning: Removing some weights of the mode. The core idea is to reduce model size
* Quantization: Use lower precision FP for computing in the middle steps. 
* Distillation: Transferring knowledge from larger models to small ones so that student models could learn faster and better. 
* MoEfication:  Partitioning models as different experts and only activating some parts of them when having an input. 

## Model Selection:

* D-0: GPT2-medium
* E-D: BART-large
* E-0:BERT-large

At first I chose GPT-2-xl for the Decoder only type. GPT2-XL works well for benchmark testing, but it's too big to store on the disk on Argonne's personal directory. So we use GPT-2 medium instead.

For the E-D and E-o type I used Pangu $\alpha$ 2.6B( [Intro from Huawei](https://www.huaweicloud.com/product/modelarts/pangu.html) ) and erlangshen 1.3B( [Intro from IDEA](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)) first. However, the structure of Chinese language models is really weird.  And they're not well supported by huggingface and `transformers` API. Too many bugs. Though I can sparsify them, I can’t use the unified `transformers` API to call the model for benchmark testing like sequence classification. 

## Model Visualization 

Actually, what I want to say is that I first sparsified those large Chinese language models, I found that the distribution for models having billions of parameters is kind of different from models having millions of parameters: **The developers did some sparsification for those models**. I guess it’s because otherwise  they’re too time-consuming to compute. Here I only present weights distribution of GPT-2 medium, BART-large, BERT-large. 

There are two conclusions:

* Most parameters are close to 0
* The parameter distribution of each layer is similar to the overall distribution, but higher layers tend to deviate from this pattern. 

![vis_all_params_GPT2-medium_sparsity_0%](/figs/vis_all_params_GPT2-medium_sparsity_0%.png)

## Model Sparsification & Sizes

## Results on Benchmarks





