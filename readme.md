# VulScribeR

Official repository for our paper:
> *VulScribeR: Exploring RAG-based Vulnerability Augmentation with LLMs*
## Datasets

### Primary Datasets
[Bigvul_train](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/bigvul_train.zip),
[Bigvul test](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/bigvul_test.zip),
[Bigvul_val](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/bigvul_val.zip)

[Reveal](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/reveal_ds.zip),
[Devign](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/devign_ds.zip)

### VGX and Vulgen (used as baselines)
[VGX Full dataset](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/vgx_full.zip),
[Vulgen Full dataset from VGX paper](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/vulgen_full.zip)

### Retriever's output
[All pair matchings, including for mutation and random ones for RQ2](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/Retriever_Results.zip)

### Our Generated Vulnerable Samples
[Filtered Datasets for All RQs](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/generated_filtered.rar),
[Unfiltered Datasets for All RQs](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/generated_raw.zip)\
The unfiltered dataset contains samples from the Generator and hasn't gone through the Verification phase. They also include extra metadata that shows which clean_vul pair was used for generation, plus the vul lines.

## How to use?
[See here](https://github.com/VulScribeR/VulScribeR/blob/main/code/readme.md)

## How to train DLVD models
Go to the [models](https://github.com/VulScribeR/VulScribeR/tree/main/models) directory, the readme for each model explains how to use each of the models
