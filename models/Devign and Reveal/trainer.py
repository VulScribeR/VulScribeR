import copy
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import json
from utils import debug


def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            names, graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), f1_score(all_targets, all_predictions) * 100
    pass

def get_embeddings(model, loss_function, num_batches, data_iter, after_ggnn_file):
    model.eval()
    after_ggnn = []
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            names, graph, targets = data_iter()
            targets = targets.cuda()
            embeddings = []
            #import pdb
            #pdb.set_trace()
            predictions = model(graph, cuda=True, embeddings=embeddings)
            #import pdb
            #pdb.set_trace()
            for iii,embedding in enumerate(embeddings[0]):
                obj={}
                #obj["name"] = names[iii]
                obj["target"] = int(targets[iii].tolist())
                obj["graph_feature"] = embedding
                after_ggnn.append(obj)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        f=open(after_ggnn_file, "w")
        json.dump(after_ggnn,f)
        f.close()

        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass

def get_corrects(model, loss_function, num_batches, data_iter, correct_file):
    model.eval()
    after_ggnn = []
    correct_names = []
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        all_names = []
        for _ in range(num_batches):
            names, graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            '''
            for iii,embedding in enumerate(embeddings[0]):
                obj={}
                obj["name"] = names[iii]
                obj["target"] = int(targets[iii].tolist())
                obj["graph_feature"] = embedding
                after_ggnn.append(obj)
            '''
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            all_names.extend(names)
        for iii in range(len(all_names)):
            if int(all_targets[iii]) == int(all_predictions[iii]):
                correct_names.append(all_names[iii])

        model.train()
        f=open(correct_file, "w")
        json.dump(correct_names,f,indent=4)
        f.close()

        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


def evaluate_metrics(model, loss_function, num_batches, data_iter, neg_metrics=[]):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            names, graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            #import pdb
            #pdb.set_trace()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        tp=0
        fp=0
        tn=0
        fn=0
        for i, prediction in enumerate(all_predictions):
            if prediction == 1 and all_targets[i] == 1:
                tp+=1
            if prediction == 1 and all_targets[i] == 0:
                fp+=1
            if prediction == 0 and all_targets[i] == 0:
                tn+=1
            if prediction == 0 and all_targets[i] == 1:
                fn+=1
        neg_acc = (tn+tp)/(tn+tp+fn+fp)*100
        neg_prec = (tn)/(tn+fn)*100
        neg_recall = (tn)/(tn+fp)*100
        neg_f1 = 2*neg_prec*neg_recall/(neg_prec+neg_recall)*100
        neg_metrics.extend([neg_acc,neg_prec,neg_recall,neg_f1])

        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


def train(model, dataset, max_steps, dev_every, loss_function, optimizer, save_path, log_every=50, max_patience=5):
    debug('Start Training')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0
    try:
        for step_count in range(max_steps):
            model.train()
            model.zero_grad()
            names, graph, targets = dataset.get_next_train_batch()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            if log_every is not None and (step_count % log_every == log_every - 1):
                debug('Step %d\t\tTrain Loss %10.3f' % (step_count, batch_loss.detach().cpu().item()))
            train_losses.append(batch_loss.detach().cpu().item())
            batch_loss.backward()
            optimizer.step()
            if step_count % dev_every == (dev_every - 1):
                #train_loss, train_f1 = evaluate_loss(model, loss_function, dataset.initialize_train_batch(),dataset.get_next_train_batch)

                #acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_train_batch(),dataset.get_next_train_batch)
                #debug('Train Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (acc, pr, rc, f1))
                #acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),dataset.get_next_test_batch)
                #debug('Test Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (acc, pr, rc, f1))
                #valid_f1 = f1
                #valid_f1=train_f1
                #valid_loss, valid_f1 = evaluate_loss(model, loss_function, dataset.initialize_test_batch(),
                #                                     dataset.get_next_test_batch)
                #if True or valid_f1 > best_f1:
                #patience_counter = 0
                #best_f1 = valid_f1
                #best_model = copy.deepcopy(model)
                _save_file = open(save_path + '-model.bin'+str(step_count), 'wb')
                torch.save(model, _save_file)
                _save_file.close()
                #else:
                #    patience_counter += 1
                #debug('Step %d\t\tTrain Loss %10.3f\tTest Loss%10.3f\tf1: %5.2f\tPatience %d' % (
                #    step_count, np.mean(train_losses).item(), valid_loss, valid_f1, patience_counter))
                #debug('Step %d\t\tTrain Loss %10.3f\tTrain Loss%10.3f\tf1: %5.2f\tPatience %d' % (step_count, np.mean(train_losses).item(), train_loss, train_f1, patience_counter))
                debug('Step %d' % (step_count))
                debug('=' * 100)
                train_losses = []
                #if patience_counter == max_patience:
                if step_count>max_steps:
                    break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')

    #if best_model is not None:
    #    _save_file = open(save_path + '-model.bin', 'rb')
    #    model = torch.load(_save_file)
    #    _save_file.close()
        #model.load_state_dict(best_model)
    #_save_file = open(save_path + '-model.bin', 'wb')
    #torch.save(model, _save_file)
    #_save_file.close()
    acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_train_batch(),
                                       dataset.get_next_train_batch)
    debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    debug('=' * 100)
    return model
    

