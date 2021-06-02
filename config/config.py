import cornac
import os
import argparse

"""
(0.38491889991116923, 0.20341320353025805, 0.9640238789214913) # bpr 128
(0.37798884045505843, 0.19964004476210226, 0.9543370175183646) # bpr 256
(0.7000577042903006, 0.6286339475068515, 0.9181019389876811) # wbpr 128
(0.696765565186481, 0.6236125356834415, 0.9203812003227483) # wbpr 512
"""

parser = argparse.ArgumentParser()

parser.add_argument(
        "--work_path",
        default="/home/kevin/nut/MITB/recommendor/pj1",
        type=str,
        required=False,
        help="working path",
    )
parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='verbose or not, default False')
parser.add_argument(
        '--threshold',
        type=int,
        default=50,
        help='metrics threshold, default 50')
parser.add_argument(
        '--seed',
        type=int,
        default=7,
        help='random seed, default 7')
parser.add_argument(
        "--train_path",
        default='/data/cs608_ip_train_v3.csv',
        type=str,
        required=False,
        help="train data path",
    )
parser.add_argument(
        "--dev_path",
        default='/data/cs608_ip_train_v3.csv',
        type=str,
        required=False,
        help="dev data path",
    )
parser.add_argument(
        "--checkout_path",
        default='/checkout',
        type=str,
        required=False,
        help="dev data path",
    )
args = parser.parse_args()

threshold = args.threshold
epsilon = 1e-7
work_path = args.work_path
VERBOSE=args.verbose
SEED=args.seed

train_data_path = work_path + args.train_path
dev_data_path = work_path + args.dev_path

test_path = work_path + '/submit/kaman.txt'
checkout_path= work_path+ args.checkout_path

epochs = 50
num_item = 17946