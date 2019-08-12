import argparse
import pickle
import os


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Shorten link prediction dataset (for test purposes)',
        usage='shorten_dataset.py [<args>]'
    )
    parser.add_argument('-i', '--input', type=str, default=None)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-p', '--portion', type=int, default=30)

    return parser.parse_args(args)


def main(args):
    """
    read input data from pickle files,
    shorten them and create output (pickle files)
    """

    if args.input is None or args.output is None:
        raise ValueError('You have to specify input and output paths.')

    if args.portion < 1 or args.portion > 100:
        raise ValueError('portion has to be a value between 1 and 100.')

    input_path = args.input
    output_path = args.output
    portion = args.portion

    path_train = os.path.join(input_path, 'train.pickle')
    with open(path_train, 'rb') as handle:
        train_triples = pickle.load(handle)

    path_valid = os.path.join(input_path, 'valid.pickle')
    with open(path_valid, 'rb') as handle:
        valid_triples = pickle.load(handle)

    path_test = os.path.join(input_path, 'test.pickle')
    with open(path_test, 'rb') as handle:
        test_triples = pickle.load(handle)

    train_triples = train_triples[:round(len(train_triples) * (portion/100))]
    valid_triples = valid_triples[:round(len(valid_triples) * (portion/100))]
    test_triples = test_triples[:round(len(test_triples) * (portion/100))]

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    path_train_o = os.path.join(output_path, 'train.pickle')
    with open(path_train_o, "wb") as handle:
        pickle.dump(train_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    path_valid_o = os.path.join(output_path, 'valid.pickle')
    with open(path_valid_o, "wb") as handle:
        pickle.dump(valid_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    path_test_o = os.path.join(output_path, 'test.pickle')
    with open(path_test_o, "wb") as handle:
        pickle.dump(test_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(parse_args())
