import argparse
from utils import preprocessing


parser = argparse.ArgumentParser(description='Data preprocessing')
parser.add_argument('--train_path', default='./data/train.tsv', type=str, help='Path to train dataframe.')
parser.add_argument('--test_path', default='./data/test.tsv', type=str, help='Path to test dataframe.')
parser.add_argument('--test_save_path', default='./data/test_proc.tsv', type=str,
                    help='Path to save preprocessed test dataframe')
parser.add_argument('--chunksize', default=100, type=int, help='Preprocessing batch size')

args = parser.parse_args()
preprocessing(args.train_path, args.train_path, args.test_save_path, args.chunksize)
