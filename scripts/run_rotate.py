import argparse
import json
import logging
import os
import pickle
import mlflow
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.models.model import KGEModel
from src.data.dataloader import TrainDataset, BidirectionalOneShotIterator
from mlflow import log_metric, log_param


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='RotatE', type=str)
    parser.add_argument('-de', '--double_entity_embedding',
                        action='store_true')

    parser.add_argument('-n', '--neg_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=1000, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling',
                        action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0,
                        type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int,
                        help='valid/test batch size')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int,
                        help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int,
                        help='valid/test log every xx steps')

    return parser.parse_args(args)


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )


def save_scores(scores, scores_bias, save_path):
    scores_path = os.path.join(save_path, 'scores.pickle')
    with open(scores_path, 'wb') as handle:
        pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    scores_bias_path = os.path.join(save_path, 'scores_bias.pickle')
    with open(scores_bias_path, 'wb') as handle:
        pickle.dump(scores_bias, handle, protocol=pickle.HIGHEST_PROTOCOL)


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    if args.do_train:
        # log_file
        # = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        # log_file
        # = os.path.join(args.save_path or args.init_checkpoint, 'test.log')
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metric_with_logging(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f', mode, metric, step, metrics[metric]) # noqa


def visualize_loss(pos_sample_loss, neg_sample_loss, loss, save_path):
    step_count = range(1, len(loss)+1)
    plt.figure()
    plt.title('training loss')
    plt.plot(step_count, pos_sample_loss, 'g-')
    plt.plot(step_count, neg_sample_loss, 'r-')
    plt.plot(step_count, loss, 'b-')
    plt.legend(['pos_sample_loss', 'neg_sample_loss',
                'loss (avg of pos & neg)'])
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig(os.path.join(save_path, 'train_loss.png'))

    step_count = range(1, min(len(loss)+1, 11001))
    plt.figure()
    plt.title('training loss')
    plt.plot(step_count, pos_sample_loss[:11000], 'g-')
    plt.plot(step_count, neg_sample_loss[:11000], 'r-')
    plt.plot(step_count, loss[:11000], 'b-')
    plt.legend(['pos_sample_loss', 'neg_sample_loss',
                'loss (avg of pos & neg)'])
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig(os.path.join(save_path, 'first_steps_train_loss.png'))


def visualize_metrics(metrics, save_path, valid_steps):
    hits1 = []
    hits3 = []
    hits10 = []
    # mr = []
    mrr = []
    for i in range(len(metrics)):
        # mr.append(metrics[i]['MR'])
        mrr.append(metrics[i]['MRR'])
        hits1.append(metrics[i]['HITS@1'])
        hits3.append(metrics[i]['HITS@3'])
        hits10.append(metrics[i]['HITS@10'])

    step_count = [x*valid_steps for x in range(1, len(mrr)+1)]
    plt.figure()
    plt.title('validation metrics')
    # plt.plot(step_count, mr, 'b-')
    plt.plot(step_count, mrr)
    plt.plot(step_count, hits1)
    plt.plot(step_count, hits3)
    plt.plot(step_count, hits10)
    plt.legend(['MRR', 'HITS@1', 'HITS@3', 'HITS@10'])  # 'MR'
    plt.xlabel('step')
    plt.savefig(os.path.join(save_path, 'valid_metrics.png'))


def train_loop(args, kge_model, optimizer, train_iterator, valid_triples,
               all_true_triples):
    init_step = 1  # init_steps = 0 (and not: add 1 to args.max_steps)
    training_logs = []
    training_logs_all = []
    valid_logs = []
    warm_up_steps = args.max_steps // 2
    current_learning_rate = args.learning_rate

    for step in range(init_step, args.max_steps+1):
        log = kge_model.train_step(kge_model, optimizer, train_iterator,
                                   args)
        training_logs.append(log)

        if step >= warm_up_steps:
            current_learning_rate = current_learning_rate / 10
            logging.info('Change learning_rate to %f at step %d', current_learning_rate, step)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
            warm_up_steps = warm_up_steps * 3

        if step % args.save_checkpoint_steps == 0:
            save_variable_list = {
                'step': step,
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
            }
            save_model(kge_model, optimizer, save_variable_list, args)

        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in
                                       training_logs])/len(training_logs)
            log_metric_with_logging('Training average', step, metrics)
            training_logs_all.extend(training_logs)
            training_logs = []

        if args.do_valid and step % args.valid_steps == 0:
            logging.info('Evaluating on Valid Dataset...')
            metrics, _, _, _ = kge_model.test_step(kge_model, valid_triples,
                                                   all_true_triples, args)
            valid_logs.append(metrics)
            log_metric_with_logging('Valid', step, metrics)

    pos_sample_loss = [log["pos_sample_loss"]
                       for log in training_logs_all]
    neg_sample_loss = [log["neg_sample_loss"]
                       for log in training_logs_all]
    loss = [log["loss"] for log in training_logs_all]
    visualize_loss(pos_sample_loss, neg_sample_loss, loss, args.save_path)

    visualize_metrics(valid_logs, args.save_path, args.valid_steps)
    save_variable_list = {
        'step': step,
        'current_learning_rate': current_learning_rate,
        'warm_up_steps': warm_up_steps
    }
    save_model(kge_model, optimizer, save_variable_list, args)


def load_data(data_path):
    path_entity2id = os.path.join(data_path, 'entity_to_id.pickle')
    with open(path_entity2id, "rb") as handle:
        entity2id = pickle.load(handle)

    path_relation2id = os.path.join(data_path, 'relation_to_id.pickle')
    with open(path_relation2id, "rb") as handle:
        relation2id = pickle.load(handle)

    path_train = os.path.join(data_path, 'train.pickle')
    with open(path_train, 'rb') as handle:
        train_triples = pickle.load(handle)

    path_valid = os.path.join(data_path, 'valid.pickle')
    with open(path_valid, 'rb') as handle:
        valid_triples = pickle.load(handle)

    path_test = os.path.join(data_path, 'test.pickle')
    with open(path_test, 'rb') as handle:
        test_triples = pickle.load(handle)

    return entity2id, relation2id, train_triples, valid_triples, test_triples


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be chosen.')

    # if args.do_train and args.save_path is None:
    #     raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    args.hidden_dim = int(round(args.hidden_dim))
    loaded_data = load_data(args.data_path)
    entity2id = loaded_data[0]
    relation2id = loaded_data[1]
    train_triples = loaded_data[2]
    valid_triples = loaded_data[3]
    test_triples = loaded_data[4]

    nentity = len(entity2id)
    nrelation = len(relation2id)
    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s', args.model)
    logging.info('Data Path: %s', args.data_path)
    logging.info('# entity: %d', nentity)
    logging.info('# relation: %d', nrelation)

    logging.info('# train: %d', len(train_triples))
    logging.info('# valid: %d', len(valid_triples))
    logging.info('# test: %d', len(test_triples))

    all_true_triples = np.concatenate([train_triples, valid_triples,
                                       test_triples], axis=0)

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        # double_relation_embedding=args.double_relation_embedding
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s',
                     name,
                     str(param.size()),
                     str(param.requires_grad))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation,
                         args.neg_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation,
                         args.neg_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head,
                                                      train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )

    logging.info('Randomly Initializing %s Model...', args.model)
    init_step = 0
    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d', init_step)
    # logging.info('learning_rate = %d', current_learning_rate)
    logging.info('batch_size = %d', args.batch_size)
    logging.info('negative_adversarial_sampling = %d', args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d', args.hidden_dim)
    logging.info('gamma = %f', args.gamma)
    logging.info('negative_adversarial_sampling = %s', str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f', args.adversarial_temperature)

    valid_metrics = dict()
    # Set valid dataloader as it would be evaluated during training
    if args.do_train:
        # Training Loop
        start_training = time.time()
        train_loop(args, kge_model, optimizer, train_iterator, valid_triples,
                   all_true_triples)
        end_training = time.time()
        logging.info('total training time: %d',round(end_training - start_training))

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        valid_metrics, _, _, _ = kge_model.test_step(kge_model, valid_triples,
                                                     all_true_triples, args)
        log_metric_with_logging('Valid', step, valid_metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics, _, _, _ = kge_model.test_step(kge_model,
                                               test_triples,
                                               all_true_triples,
                                               args)

        log_metric_with_logging('Test', step, metrics)

        # log_metric and log_param (mlflow)
        with mlflow.start_run():
            log_metric('Mean Rank', metrics['MR'])
            log_metric('Mean Reciprocal Rank', metrics['MRR'])
            log_metric('Hits at 1', metrics['HITS@1'])
            log_metric('Hits at 3', metrics['HITS@3'])
            log_metric('Hits at 10', metrics['HITS@10'])
            log_param('Batch Size', args.batch_size)
            log_param('Negative Sample Size', args.neg_sample_size)
            log_param('Hidden Dim', args.hidden_dim)
            log_param('Gamma', args.gamma)
            log_param('Adversarial Temperature', args.adversarial_temperature)
            log_param('Learning Rate', args.learning_rate)
            log_param('Max Steps', args.max_steps)

        # save_scores(scores, scores_bias, args.save_path)

    return valid_metrics  # just for hyperparameter tuning


if __name__ == '__main__':
    start = time.time()
    main(parse_args())
    end = time.time()
    logging.info('total time: %d', round(end-start))
