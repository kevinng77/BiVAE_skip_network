import argparse

"""
(0.3396387325165736, 0.18592470825151217, 0.7921542766832573) # bpr 128
(0.3303023419089757, 0.18160128826536834, 0.7694795583677434) # bpr 256
(0.5503921443471618, 0.5199370169318711, 0.6500497602928873) # wbpr 128
(0.5914794038835678, 0.5531050101652778, 0.7085162250915917) # wbpr 256
(0.5504878424202072, 0.5176797798555474, 0.6557992454716938) # wbpr 512
(0.14772097122738445, 0.07328380031837196, 0.371795722479392) # kaman.txt
"""

# default argument
epsilon = 1e-7

# argument
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
parser.add_argument('--true_threshold',type=int,default=2,help='threshold for label ground truth, default 2')
parser.add_argument('--num_neg',type=int,default=10,help='number of negative samples, default 10')
parser.add_argument('--fre_factor',type=float,default=0.75,help='default 0.75')
parser.add_argument('--rank_factor',type=float,default=1.0,help='default 0.75')

args = parser.parse_args()

# file path argument
work_path = args.work_path
checkout_path = work_path + args.checkout_path
train_data_path = work_path + args.train_path
dev_data_path = work_path + args.dev_path
result_path = work_path + '/submit/WBPR(K=512).txt'
col_name = ['user_id', 'item_id', 'rating']
# debug argument
VERBOSE = args.verbose
SEED = args.seed

#
threshold = args.threshold
epochs = 50
num_item = 17946
num_user = 38609

num_neg = args.num_neg
fre_factor = args.fre_factor
rank_factor = args.rank_factor
true_threshold = args.true_threshold  # rate greater than true threshold will be label as 1
