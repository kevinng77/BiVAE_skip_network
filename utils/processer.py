import random
import numpy as np
import pandas as pd


def load_data(train_path, dev_path):
    df_train = pd.read_csv(train_path)
    df_eval = pd.read_csv(dev_path)
    return df_train, df_eval


def gen_neg_weight(df_train, fre_factor, rank_factor, num_item):
    population = np.arange(1, num_item + 1)
    item_fre = df_train.groupby('item_id').count().user_id.values
    item_avg_rank = df_train.groupby('item_id').mean().rating.values
    item_fre = item_fre ** fre_factor
    item_avg_rank = item_avg_rank ** rank_factor
    # high fre = high neg sample rate
    # high rate = low neg sample rate
    # fre 10 rate 5=2, fre 3 rate 1=3
    item_neg_weight = item_fre + 5 ** rank_factor / item_avg_rank
    return item_neg_weight, population


def user_neg_weight(item_neg_weight, rated_items):
    rated_items_idx = [x - 1 for x in rated_items]
    item_neg_weight[rated_items_idx] = 0
    return item_neg_weight


def gen_neg_sample_mbpr_v2(df, fp, user_id, population, item_neg_weight,
                           true_threshold, num_neg, col_name):
    """
    1. for user selected items, high rate pred> low rate pred
    2. for user selected items, rate above true_threshold pred > no rate item
    3. Modified dataprocessor with out pandas output. Speed up processing time.
    :param df: [pd.DataFrame] raw training dataset
    :param num_neg: [int] number of negative samples per user-rated item
    :return: generate negative samples for user[id]. this user might have different items
    """
    df.columns = col_name
    df_rated_item = df[df.user_id == user_id]
    high_rated_items = list(df_rated_item[df_rated_item.rating >= true_threshold].item_id)
    neg_weight = user_neg_weight(item_neg_weight, high_rated_items)

    for _, row in df_rated_item.iterrows():
        iid = row['item_id']
        rating = row['rating']
        neg_items = list(df_rated_item[df_rated_item.rating < rating].item_id)
        if rating >= true_threshold:
            neg_items.extend(random.choices(population, weights=neg_weight, k=num_neg))
        record_list = [[user_id, iid, neg_items[i]] for i in range(len(neg_items))]
        for line in record_list:
            fp.write(",".join([str(x) for x in line]) + '\n')


def gen_neg_sample_mbpr(df, mbpr_col_name, user_id, population, item_neg_weight, true_threshold, num_neg, col_name):
    """
    1. select based on the rated frequency of items
    2. select based on the average rating of items
    3.
    :param df: [pd.DataFrame] raw training dataset
    :param num_neg: [int] number of negative samples per user-rated item
    :return: generate negative samples for user[id]. this user might have different items
    """
    df.columns = col_name
    user_df = pd.DataFrame(columns=mbpr_col_name)
    df_rated_item = df[df.user_id == user_id]
    high_rated_items = list(df_rated_item[df_rated_item.rating >= true_threshold].item_id)
    neg_weight = user_neg_weight(item_neg_weight, high_rated_items)
    for _, row in df_rated_item.iterrows():
        iid = row['item_id']
        rating = row['rating']
        neg_items = list(df_rated_item[df_rated_item.rating < rating].item_id)
        if rating >= true_threshold:
            neg_items.extend(random.choices(population, weights=neg_weight, k=num_neg))
        record_list = [[user_id, iid, neg_items[i]] for i in range(len(neg_items))]
        user_df = user_df.append(pd.DataFrame(record_list, columns=mbpr_col_name))
    return user_df


def gen_train_dataset_v2(file_name, df_train, num_neg, true_threshold, mbpr_col_name, fre_factor,
                         rank_factor, num_item, col_name, VERBOSE=1):
    """
    data set contain 3 items: user id, item id, rank
    item id: high quality positive item and negative item were selected for users.
    rank: positive items rank 1, negative items rank 0.
    Binary classification task
    """
    user_ids = df_train[mbpr_col_name[0]]
    user_list = list(set(user_ids))
    item_neg_weight, population = gen_neg_weight(df_train, fre_factor, rank_factor, num_item)
    count = 0
    with open(file_name, "w")as fp:
        fp.write(",".join(mbpr_col_name) + '\n')
        for user_id in user_list:
            if VERBOSE and (count % 5000 == 0):
                print(f'{count}/{len(user_list)}')
            gen_neg_sample_mbpr_v2(df_train, fp, user_id, population, item_neg_weight, true_threshold, num_neg,
                                   col_name),
            count += 1


def gen_train_dataset(df_train, num_neg, true_threshold, mbpr_col_name, fre_factor, rank_factor, num_item, col_name,
                      VERBOSE=1):
    """
    data set contain 3 items: user id, item id, rank
    item id: high quality positive item and negative item were selected for users.
    rank: positive items rank 1, negative items rank 0.
    Binary classification task
    """
    user_ids = df_train[mbpr_col_name[0]]
    user_list = list(set(user_ids))
    item_neg_weight, population = gen_neg_weight(df_train, fre_factor, rank_factor, num_item)
    df_main = pd.DataFrame(columns=mbpr_col_name)
    count = 0
    for user_id in user_list:
        if VERBOSE and (count % 5000 == 0):
            print(f'{count}/{len(user_list)}')
        df_main = df_main.append(
            gen_neg_sample_mbpr(df_train, mbpr_col_name, user_id, population, item_neg_weight,
                                true_threshold, num_neg,col_name),
            ignore_index=True)
        count += 1
    return df_main
