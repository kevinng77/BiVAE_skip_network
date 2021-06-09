import argparse

import config.config

col_name = ['user_id', 'item_id', 'rating']
mbpr_col_name = ['user_id', 'pos_item_id', 'neg_item_id']


# argument
parser = argparse.ArgumentParser()
parser.add_argument("--work_path", default="/home/kevin/nut/MITB/recommendor/pj1", type=str, required=False,
                    help="working path", )

parser.add_argument('--verbose', action='store_true', default=False,
                    help='verbose or not, default False')

parser.add_argument("--train_data_path", default='/data/Modify_train_3_10.csv', type=str, required=False,
                    help="train data path, after processing as MBPR dataset", )

parser.add_argument("--raw_data_path", default='/data/cs608_ip_train_v3.csv', type=str, required=False,
                    help="raw train data path with col user_id, item_id, rating", )

parser.add_argument("--dev_path", default='/data/cs608_ip_probe_v3.csv', type=str, required=False,
                    help="dev data path")

parser.add_argument("--checkout_path", default='/checkout', type=str, required=False,
                    help="checkout folder path")

parser.add_argument("--load_weight", default='0', type=str, required=False,
                    help="load model weight")

parser.add_argument('--seed', type=int, default=7, help='random seed, default 7')
parser.add_argument('--threshold', type=int, default=50, help='metrics threshold, default 50')

# param for data processing
parser.add_argument('--true_threshold', type=int, default=2,
                    help='threshold for label ground truth, default 2')
parser.add_argument('--num_neg', type=int, default=10, help='number of negative samples, default 10')
parser.add_argument('--fre_factor', type=float, default=0.75, help='default 0.75')
parser.add_argument('--rank_factor', type=float, default=1.25, help='default 1.25')
parser.add_argument('--dtype', type=str, default="train",
                    help='process data type, "train" or "merge" , default "train"')

# param for train
parser.add_argument('--num_item', type=int, default=17946, help='number of total_item')
parser.add_argument('--num_user', type=int, default=38609, help='number of total user')
parser.add_argument('--k', type=int, default=20, help='number of factor dimension, default 50')
parser.add_argument('--epochs', type=int, default=200, help='number of training epoch, default 200')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default 1e-4')
parser.add_argument('--lambda_reg', type=float, default=1e-3, help='regularization weight, default 1e-3')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum beta, default 0.9')
parser.add_argument('--num_threads', type=int, default=4, help='number of threads, default 4')
parser.add_argument('--batch_size', type=int, default=128, help=' default 128')
parser.add_argument('--encoder_structure', type=int, nargs='+',default=[128], help='[40]')
parser.add_argument('--decoder_structure', type=int, nargs='+',default=[40], help='[40]')


args = parser.parse_args()

# file path argument
work_path = args.work_path
checkout_path = work_path + args.checkout_path
raw_data_path = work_path + args.raw_data_path  # DATA before processing
train_data_path = work_path + args.train_data_path  # MBPR processed data
dev_data_path = work_path + args.dev_path  # PATH for metrics evaluate
# load_weight = work_path + args.load_weight

# debug argument
VERBOSE = args.verbose
SEED = args.seed

threshold = args.threshold
num_item = args.num_item
num_user = args.num_user

# data processing argument
num_neg = args.num_neg
fre_factor = args.fre_factor
rank_factor = args.rank_factor
true_threshold = args.true_threshold  # rate greater than true threshold will be label as 1
dtype = args.dtype

# training variable
k = args.k
epochs = args.epochs
lr = args.lr
lambda_reg = args.lambda_reg
momentum = args.momentum
num_threads = args.num_threads
batch_size = args.batch_size
encoder_structure=args.encoder_structure
decoder_structure=args.decoder_structure
