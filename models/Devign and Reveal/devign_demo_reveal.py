import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from trainer import train, evaluate_metrics, get_embeddings
from utils import tally_param, debug


if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='target')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=169)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=128)
    args = parser.parse_args()

    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'processed.bin')
    if os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    else:
        dataset = DataSet(train_src=['train.json.shard1','train.json.shard2','train.json.shard3','train.json.shard4','train.json.shard5', 'train.json.shard6', 'train.json.shard7','train.json.shard8', 'train.json.shard9','train.json.shard10'],
                          valid_src=None,
                          test_src=['reveal.json.shard1', 'reveal.json.shard2', 'reveal.json.shard3', 'reveal.json.shard4'],
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()
    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    vulsamples=0
    for train_example in dataset.train_examples:
        if train_example.target == 1:
            vulsamples += 1
    print("train:", len(dataset.train_examples), vulsamples)
    vulsamples=0
    for test_example in dataset.test_examples:
        if test_example.target == 1:
            vulsamples += 1
    print("test:", len(dataset.test_examples), vulsamples)
 
    if args.model_type == 'ggnn':
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    else:
        model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    model.cuda()
    loss_function = BCELoss(reduction='sum')
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    model.train()
    #model.load_state_dict(torch.load("./models/reveal_gen/GGNNSumModel-model.bin"))
    #model = torch.load("./models/real_world_ori/GGNNSumModel-model.bin")
    model = train(model=model, dataset=dataset, max_steps=2000, dev_every=128,
          loss_function=loss_function, optimizer=optim,
          save_path=model_dir + '/GGNNSumModel', max_patience=100, log_every=None)
    neg_metrics = []
    acc, prec, rec, f1 = evaluate_metrics(model, loss_function,  dataset.initialize_test_batch(), dataset.get_next_test_batch, neg_metrics)
    #acc, prec, rec, f1 = get_embeddings(model, loss_function,  dataset.initialize_test_batch(), dataset.get_next_test_batch,"real_world_ori_test_after_ggnn.json")


    print("testing result:", acc,prec,rec,f1)
    print("neg testing result:", neg_metrics)

