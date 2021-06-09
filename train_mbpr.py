import numpy as np
import pandas as pd
from config import config
from models.modifybpr import MBPR
from utils import metrics
import json


def train(df, k, max_iter, lr, reg, momentum, verbose,
          num_users=config.num_user,num_threads=config.num_threads ,num_items=config.num_item):
    model_name = f"MBPR_T{config.true_threshold}_N{config.num_neg}"
    model = MBPR(k=k, name=model_name, max_iter=max_iter, learning_rate=lr,
                 lambda_reg=reg, momentum=momentum, verbose=verbose)
    try:
        model.load(config.checkout_path)
    except:
        pass
    print("training:", model.name)
    model.fit(train_set = df, num_users= num_users,num_items= num_items,num_thread=num_threads ,val_set=None)
    return model


def eval(checkout_path, ground_truth, recommend_result, threshold, num_item,loss):
    NDCG, NCRR, RECALL = metrics.evaluate_from_file(
        recommend_result, ground_truth, threshold, num_item)
    log = f"{NDCG:.4f}\t{NCRR:.4f}\t{RECALL:.4f}" \
          f"\tk: {config.k}\tT: {config.true_threshold}\tneg: {config.num_neg}\tlr:" \
          f"{config.lr}\tloss:{loss:.5f}\treg:{config.lambda_reg*100:.2f}%\t{ground_truth[-21:]}\n"
    metrics.save_metrics(checkout_path, log)
    return NDCG, NCRR, RECALL


def print_score(model,recommend_path,dev_data):
    score = eval(config.checkout_path, dev_data, recommend_path,
                 config.threshold, config.num_item,model.losses[-1])
    print([round(x, 5) for x in score], model.name,dev_data[-25:])


if __name__ == '__main__':
    np.random.seed(config.SEED)
    verbose = config.VERBOSE
    df_train = pd.read_csv(config.train_data_path)  # load training data
    model = train(df_train, config.k, config.epochs, config.lr, config.lambda_reg, config.momentum, verbose)
    param_log_name = f"{model.name}_{config.train_data_path[-9:-4]}.json"
    recommend_path = model.save(config.checkout_path, param_log_name)
    metrics.save_metrics(config.checkout_path, '-'*75+'\n')
    print_score(model, recommend_path, config.dev_data_path)
    print_score(model, recommend_path, config.raw_data_path)
    print_score(model, recommend_path, config.work_path+"/data/train_high_fre_user.csv")
    print_score(model, recommend_path, config.work_path+"/data/test_high_fre_user.csv")

