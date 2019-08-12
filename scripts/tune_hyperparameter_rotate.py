import run_rotate
import os
import argparse
from collections import namedtuple

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default=None)
    parser.add_argument('-s', '--save_path', type=str, default=None)

    return parser.parse_args(args)


Arguments = namedtuple('Arguments',
                       ['cuda', 'do_train', 'do_valid', 'do_test',
                        'data_path', 'model', 'double_entity_embedding',
                        'neg_sample_size', 'hidden_dim', 'gamma', 'batch_size',
                        'adversarial_temperature','negative_adversarial_sampling',
                        'test_batch_size', 'learning_rate', 'cpu_num', 'save_path',
                        'max_steps', 'save_checkpoint_steps', 'valid_steps', 'log_steps',
                        'test_log_steps'])

class Arguments:
    def __init__(self):
        self.cuda = None
        self.do_train = None
        self.do_valid = None
        self.do_test = None
        self.data_path = None
        self.model = None
        self.double_entity_embedding = None
        self.neg_sample_size = None
        self.hidden_dim = None
        self.gamma = None
        self.batch_size = None
        self.adversarial_temperature = None
        self.negative_adversarial_sampling = None
        self.test_batch_size = None
        self.learning_rate = None
        self.cpu_num = None
        self.save_path = None
        self.max_steps = None
        self.save_checkpoint_steps = None
        self.valid_steps = None
        self.log_steps = None
        self.test_log_steps = None

def create_args():
    args = Arguments()
    args.cuda=True
    args.do_train=True
    args.do_valid=True
    args.do_test=False
    args.data_path=None
    args.model='RotatE'
    args.double_entity_embedding=True
    args.neg_sample_size=256
    args.hidden_dim=1000
    args.gamma=9.0
    args.negative_adversarial_sampling=True
    args.adversarial_temperature=1.219
    args.batch_size=1024
    test_batch_size=16
    args.learning_rate=0.0000254
    args.cpu_num=10
    args.save_path=None
    args.max_steps=100000
    args.save_checkpoint_steps=10000
    args.valid_steps=10000
    args.log_steps=100
    args.test_log_steps=1000
    return args


def black_box_function(batch_size, neg_sample_size, max_steps):
    """Function with unknown internals we wish to maximize.
    """
    args = parse_args()
    additional_args = create_args()
    additional_args.save_path = '../models/tmp'
    additional_args.save_path = args.save_path
    additional_args.data_path = args.data_path
    
    info = [batch_size, neg_sample_size, max_steps]
    output_path = os.path.join(additional_args.save_path, str(info))
    additional_args.save_path = output_path

    if batch_size is not None:
        additional_args.batch_size = int(round(batch_size))
    if neg_sample_size is not None:
        additional_args.neg_sample_size = int(round(neg_sample_size))
    if max_steps is not None:
        additional_args.max_steps = int(round(max_steps))

    args = additional_args
    metrics = run_rotate.main(args)

    combined_metrics = metrics['MRR'] + metrics['HITS@1'] + metrics['HITS@10']
    return combined_metrics


def main():
    # Bounded region of parameter space
    pbounds = {'batch_size': (768, 1280),
               'neg_sample_size': (128, 384),
               'max_steps': (1000, 100000)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    logger = JSONLogger(path='../models/tmp/logs.json')
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    # try to maximize metrics, results will be saved in models/tmp/logs.json
    optimizer.maximize(init_points=1, n_iter=2)

    print(optimizer.max)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))


if __name__ == '__main__':
    main()
