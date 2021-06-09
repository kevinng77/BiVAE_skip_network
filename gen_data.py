import random
import numpy as np
import pandas as pd
from config import config
from utils import processer
import time

if __name__ == '__main__':
    time1 = time.time()
    df_train, df_dev = processer.load_data(config.raw_data_path, config.dev_data_path)
    file_path = config.work_path + f'/data/Modify_{config.dtype}_{config.true_threshold}_{config.num_neg}.csv'
    print(f"threshold: {config.true_threshold},num_neg: {config.num_neg}")
    processer.gen_train_dataset_v2(
        file_path,df_train, config.num_neg, config.true_threshold, config.mbpr_col_name,
        config.fre_factor, config.rank_factor, config.num_item, config.col_name, config.VERBOSE)
    print(time.time()-time1)
