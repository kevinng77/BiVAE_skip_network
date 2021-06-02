import os
import pandas as pd
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

import cornac
from cornac.utils import cache
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import MF, BPR, WMF
from config import config
import tensorflow as tf
print(f"System version: {sys.version}")
print(f"Cornac version: {cornac.__version__}")
print(f"Tensorflow version: {tf.__version__}")
import models

SEED = 42
VERBOSE = False


if __name__ == '__main__':
    df_train = pd.read_csv(config.path_train)
    df_probe = pd.read_csv(config.path_dev)
    train_data = cornac.data.Dataset.from_uir(df_train.itertuples(index=False))
    val_data = cornac.data.Dataset.from_uir(df_probe.itertuples(index=False))
    wbpr1 = models.Modify_BPR.WBPR(k=50, max_iter=700, learning_rate=3e-3, lambda_reg=0.001,
                                              verbose=config.VERBOSE, seed=config.SEED, name=f"WBPR(K={50})").fit(train_data)
