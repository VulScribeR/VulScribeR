# Official Source Code of VulScribeR

## Important Notes
- Joern requires building (before executing ./joern-parse "some dir") and the repo does not contain the binaries and jar files. Alternatively, the fully built version can be downloaded from [here](https://github.com/VulScribeR/VulScribeR/releases/download/Release/joern.zip)
- For the filtering step (Joern), the [Reveal](https://github.com/VulScribeR/VulScribeR/blob/main/requirements-reveal-env.txt) environment can be used, and for the other steps the [Default/Linevul](https://github.com/VulScribeR/VulScribeR/blob/main/requirements-linevul-env.txt) environment can be used.
  
## Replication Steps 
0) Clustering Vulnerable samples into 5 clusters -> use kmeans.ipynb
1) Retrieval phase ->  RAG/com/anon/RAG/Main.java
2) Clustered Sampling, Formulation and Generation -> use generation.ipynb
3) Filtering the Generated samples, adding them to Devign dataset, and adding the extra items for keeping the ratio -> data_selection.ipynb 
4) Diversity calculation using entropy -> diversity_calculation.ipynb

Finally, (or after step 3) the augmented dataset will be available in ./container_data/train (for Devign and Reveal) and in the form of a jsonl file for training Linevul (step 4 uses the same jsonl file)

