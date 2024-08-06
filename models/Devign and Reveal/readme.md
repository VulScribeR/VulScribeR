# Help


## Devign
0. run "conda activate reveal" for preprocessing
1. Prepare your samples in a folder (e.g., demo)
In the folder, label the vulnerable samples with _1.c, and normal samples with _0.c
2. Put your folder to ./preprocessing/joern/folder_name. Then go to ./preprocessing/joern/ and run ./joern-parse ./folder_name(ensure that there is no folder named parsed)
3. Copy folder_name to ./preprocessing/data_processing/ and rename it "code"
Copy parsed/folder_name to ./preprocessing/data_processing/ and rename it "parsed"
4. Go to ./preprocessing/data_processing/ and run "python extract_slices.py" and "bash get_ggnn_input.sh".
5. Copy mydata.json.shard* to the root directory of Devign

Run "conda activate devign" to activate the Devign environment
6. Edit devign_demo.py, and edit line 50 and 52 for training and testing data. Edit line 87 for max_steps.
7. Run "bash devign_demo.sh" (also devign_r.sh and devign_b.sh train devign with reveal and Bigvul 5 times, respectively.)
## Reveal
Reveal is in the representation_learning directory. Reveal uses Devign's preprocessing step, so after doing the steps above, do the following:
0. run "conda activate devign"
1. run "sh reveal_embed.sh" to get reveal's embeddings
2. run "conda activate reveal"
3. copy the embeddings for train and test to reveal folder i.e.: run "cp embed_devign_reveal_t* representation_learning/"
4. run python representation_learning/reveal.py


## Source
(Vulgen's Artifacts)[https://zenodo.org/records/7552876]
