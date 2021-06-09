import numpy as np
import math
import pandas as pd


def get_rank(y_pred, y_true):
    assert isinstance(y_pred, np.ndarray) & isinstance(y_true, np.ndarray), \
        "Type of y_pred and y_true should be numpy"
    assert y_pred.shape == y_true.shape, \
        f"Shape of predition {y_pred.shape} do not match ground truth {y_true.shape}."
    s_pred = y_true[np.argsort(-y_pred)]
    return s_pred


def format_input(fn):
    def wrapper(cur_y_pred, cur_y_true, threshold=None, base=2, beta=1):
        sort_y = get_rank(cur_y_pred, cur_y_true)
        if "threshold" in fn.__code__.co_varnames:
            assert threshold is not None, \
                f"Argument threshold is required for {fn.__name__}"
            if "beta" in fn.__code__.co_varnames:
                return fn(sort_y, threshold, beta)
            elif "base" in fn.__code__.co_varnames:
                return fn(sort_y, threshold, base)
            else:
                return fn(sort_y, threshold)
        else:
            return fn(sort_y)

    return wrapper


@format_input
def myauc(sort_y):
    """
    seems different from sklearn roc_auc_score
    """
    num_true = np.sum(sort_y)
    num_pairs = num_true * (len(sort_y) - num_true)
    mask = sort_y == 1
    true_index = np.arange(0, len(sort_y))[mask]
    num_correct = np.sum([np.sum(sort_y[rank:] == 0) for rank in true_index])
    return num_correct / num_pairs


@format_input
def mymap(sort_y):
    mask = sort_y == 1
    point = np.arange(1, 1 + len(sort_y))[mask]
    return np.mean([np.sum(sort_y[:threshold]) / threshold for threshold in point])


@format_input
def rr(sort_y):
    for i in range(len(sort_y)):
        if sort_y[i]:
            return i + 1
    return -1


def crr(sort_y):
    mask = sort_y == 1
    true_rank = np.arange(1, 1 + len(sort_y))[mask]
    return np.sum([1 / x for x in true_rank])


@format_input
def normalized_crr(sort_y):
    if np.sum(sort_y) > 0:
        return crr(sort_y) / crr(np.ones(np.sum(sort_y)))
    else:
        return 0


@format_input
def precision(sort_y, threshold):
    """
    sort_y: np.narray, sorted ground truth y.
    threhold: int. t threhold = split in rank t and t+1
    """
    return np.sum(sort_y[:threshold]) / threshold


@format_input
def recall(sort_y, threshold):
    return np.sum(sort_y[:threshold]) / np.sum(sort_y)


@format_input
def f_beta(sort_y, threshold, beta=1):
    cur_p = np.sum(sort_y[:threshold]) / threshold
    cur_r = np.sum(sort_y[:threshold]) / np.sum(sort_y)
    return (1 + beta ** 2) * (cur_p * cur_r) / (beta ** 2 * cur_p + cur_r)


def dcg(sort_y, threshold, base=2):
    temp = sort_y[:threshold]
    mask = temp == 1
    true_rank = np.arange(1, 1 + len(temp))[mask]
    return np.sum([1 / math.log(1 + x, base) for x in true_rank])


@format_input
def ndcg(sort_y, threshold, base=2):
    cur_dcg = dcg(sort_y, threshold, base)
    idea_y = np.sort(sort_y)[::-1]
    idea_dcg = dcg(idea_y, threshold, base)
    return cur_dcg / idea_dcg


def evaluate_from_factor(u_factor, i_factor, i_bias, num_items, test_file, threshold=50, verbose=1):
    df_true = pd.read_csv(test_file)
    user_list = sorted(list(set(df_true.user_id)))
    total_recall = 0
    total_NDCG = 0
    total_NCRR = 0
    total_y_pred = np.dot(u_factor, i_factor.T) + i_bias.reshape(1, -1)
    count = 0
    for uid in user_list:
        if verbose and (count % 3000 == 0):
            print(count)
        y_pred = total_y_pred[uid - 1, :]
        # modify ground truth
        rated_item = df_true[df_true['user_id'] == uid]['item_id'].values - 1
        y_true = np.zeros(num_items).astype(int)
        y_true[rated_item] = 1
        total_recall += recall(y_pred, y_true, threshold=threshold)
        total_NCRR += normalized_crr(y_pred, y_true)
        total_NDCG += ndcg(y_pred, y_true, threshold=threshold)
        count += 1
    num_test_user = len(user_list)
    total_NCRR /= num_test_user
    total_NDCG /= num_test_user
    total_recall /= num_test_user
    return total_NDCG, total_NCRR, total_recall


def evaluate_from_file(recommend_result, ground_truth, threshold,num_item):
    df_true = pd.read_csv(ground_truth)
    num_items = num_item
    user_list = sorted(list(set(df_true.user_id)))
    uid = 1
    total_recall = 0
    total_NDCG = 0
    total_NCRR = 0

    def temp_recall(sort_y, num_true, threshold):
        return np.sum(sort_y[:threshold]) / (num_true + 1e-7)

    def temp_ndcg(sort_y, num_truth, threshold, base=2):
        cur_dcg = dcg(sort_y, threshold, base)
        # idea_y = np.sort(sort_y)[::-1]
        idea_y = np.zeros(len(sort_y))
        idea_y[:num_truth] = 1
        idea_dcg = dcg(idea_y, threshold, base)
        return cur_dcg / (idea_dcg + 1e-7)

    def temp_normalized_crr(sort_y, num_true):
        return crr(sort_y) / (crr(np.ones(num_true)) + 1e-7)

    with open(recommend_result, 'r') as fp:
        while True:
            line = fp.readline().split()
            if len(line) < 3:
                break
            # if uid % int(len(user_list) / 3) == 0:
            #     print(uid)
            if uid in user_list:
                # modify ground truth
                rated_item_idx = df_true[df_true['user_id'] == uid]['item_id'].values
                y_true = np.zeros(num_items + 1)
                y_true[rated_item_idx] = 1
                num_true = len(rated_item_idx)
                sort_y = y_true[np.array([int(x) for x in line])].astype(int)
                total_recall += temp_recall(sort_y, num_true, threshold=threshold)
                total_NCRR += temp_normalized_crr(sort_y, num_true)
                total_NDCG += temp_ndcg(sort_y, num_true, threshold=threshold)
            uid += 1
    num_test_user = len(user_list)
    total_NCRR /= num_test_user
    total_NDCG /= num_test_user
    total_recall /= num_test_user
    return total_NDCG, total_NCRR, total_recall


def save_metrics(checkout_path, log):
    file_name = checkout_path + '/metrics_log.txt'
    with open(file_name, 'a')as fp:
        # fp.write("-"*60)
        fp.write(log)


if __name__ == '__main__':
    from config import config

    test_true = config.dev_data_path
    print(test_true)
    recommend_result = "../1000_0.01reg_wBPR(K=256).txt"
    print(evaluate_from_file(recommend_result, test_true, config.threshold,config.num_item), end=" ")
    print(recommend_result, " test")
    test_true = config.raw_data_path
    print(evaluate_from_file(recommend_result, test_true,config.threshold,config.num_item), end=" ")
    print(recommend_result, " train")
    test_true = "../data/train_high_fre_user.csv"
    print(evaluate_from_file(recommend_result, test_true,config.threshold,config.num_item), end=" ")
    print(recommend_result, " train_high_fre")
    test_true = "../data/test_high_fre_user.csv"
    print(evaluate_from_file(recommend_result, test_true,config.threshold,config.num_item), end=" ")
    print(recommend_result, " test_high_fre")

    # test code for evaluate_from_factor
    # checkout_path = '../checkout/model'
    # i_factor = pd.read_csv(checkout_path + '/item_MBPR_k50.csv').values
    # u_factor = pd.read_csv(checkout_path + '/user_MBPR_k50.csv').values
    # i_bias = pd.read_csv(checkout_path + '/bias_MBPR_k50.csv').values
    # print(evaluate_from_factor(u_factor, i_factor, i_bias, config.num_item, test_true, threshold=50))
