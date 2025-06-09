# VulScribeR

Official repository for our paper:
> *VulScribeR: Exploring RAG-based Vulnerability Augmentation with LLMs*

If you find this project useful in your research, please consider citing:

```
@article{daneshvar2024exploringragbasedvulnerabilityaugmentation,
      title={Exploring RAG-based Vulnerability Augmentation with LLMs}, 
      author={Seyed Shayan Daneshvar and Yu Nong and Xu Yang and Shaowei Wang and Haipeng Cai},
      year={2024},
      eprint={2408.04125},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2408.04125}, 
}
```

## Datasets

### Primary Datasets
[Bigvul_train](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/bigvul_train.zip),\
[Bigvul test](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/bigvul_test.zip),\
[Bigvul_val](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/bigvul_val.zip)\
\
[Reveal](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/reveal_ds.zip),\
[Devign](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/devign_ds.zip),\
[PrimeVul (RQ4 only)](https://github.com/DLVulDet/PrimeVul) 


### VGX and Vulgen (used as baselines)
[VGX Full dataset](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/vgx_full.zip),\
[Vulgen Full dataset from VGX paper](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/vulgen_full.zip)

### Retriever's output
[All pair matching (except for RQ4), including for mutation and random ones for RQ2](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/Retriever_Results.zip)\
[RQ4's pair matching/retriver output](https://github.com/VulScribeR/VulScribeR/releases/download/RQ4/RQ4-unfiltered.zip)

### Our Generated Vulnerable Samples
[Filtered Datasets for RQs(1-3)](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/generated_filtered.rar),\
[Unfiltered Datasets for RQs(1-3)](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/generated_raw.zip),\
[Unfiltered Datasets for RQ4](https://github.com/VulScribeR/VulScribeR/releases/download/Dataset/) \
\
The unfiltered dataset contains samples from the Generator and hasn't gone through the Verification phase. They also include extra metadata that shows which clean_vul pair was used for generation, plus the vul lines.

## How to use?
[See here](https://github.com/VulScribeR/VulScribeR/blob/main/code/readme.md)

## How to train DLVD models
Go to the [models](https://github.com/VulScribeR/VulScribeR/tree/main/models) directory, the readme for each model explains how to use each of the models
