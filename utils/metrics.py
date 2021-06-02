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
    if np.sum(sort_y)>0:
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


def test():
    y_pred = np.array([0.81, 0.75, 0.5, 0.77, 0.59, 0.72, 0.49, 0.96, 0.84, 0.91])
    y_true = np.array([0.88, 0.77, 0.07, 0.54, 0.39, 0.76, 0.44, 0.52, 0.45, 0.82]) > 0.5
    test = [
        myauc(y_pred, y_true),
        precision(y_pred, y_true, threshold=5),
        recall(y_pred, y_true, threshold=5),
        mymap(y_pred, y_true),
        f_beta(y_pred, y_true, threshold=5),
        ndcg(y_pred, y_true, threshold=5)]
    return test

def evaluate_from_factor(U,V,test_file,threshold=50):
    df_true = pd.read_csv(test_file)
    user_list = sorted(list(df_true.user_id))
    num_items = max(df_true.item_id)
    total_recall = 0
    total_NDCG = 0
    total_NCRR = 0

    for uid in user_list:
        u_vector = U[uid,:]
        y_pred = np.dot(u_vector,V.T)

        # modify ground truth
        rated_item = df_true[df_true['user_id']==uid]['item_id'].values
        y_true = np.zeros(num_items)
        y_true[rated_item] = 1

        total_recall += recall(y_pred, y_true, threshold=threshold)
        total_NCRR += normalized_crr(y_pred, y_true)
        total_NDCG += ndcg(y_pred, y_true, threshold=threshold)
    num_test_user = len(user_list)
    total_NCRR/= num_test_user
    total_NDCG /= num_test_user
    total_recall /= num_test_user
    return total_NDCG, total_NCRR, total_recall


def evaluate_from_file(file,test_file,threshold = 50):
    df_true = pd.read_csv(test_file)
    num_items = max(df_true.item_id)
    user_list = sorted(list(df_true.user_id))
    uid = 1
    total_recall = 0
    total_NDCG = 0
    total_NCRR = 0

    def temp_recall(sort_y, threshold):
        return np.sum(sort_y[:threshold]) / (np.sum(sort_y)+config.epsilon)

    def temp_ndcg(sort_y, threshold, base=2):
        cur_dcg = dcg(sort_y, threshold, base)
        idea_y = np.sort(sort_y)[::-1]
        idea_dcg = dcg(idea_y, threshold, base)
        return cur_dcg / (idea_dcg+config.epsilon)


    def temp_normalized_crr(sort_y):
        return crr(sort_y) / (crr(np.ones(np.sum(sort_y)))+config.epsilon)

    with open(file,'r') as fp:
        while True:
            line = fp.readline().split()
            if len(line) < 3:
                break
            if uid % int(len(user_list)/10)==0:
                print(uid)
            if uid in user_list:
                # modify ground truth
                rated_item_idx = df_true[df_true['user_id'] == uid]['item_id'].values-1
                y_true = np.zeros(num_items)
                y_true[rated_item_idx] = 1
                sort_y = y_true[np.array([int(x) for x in line])].astype(int)
                total_recall += temp_recall(sort_y,threshold=threshold)
                total_NCRR += temp_normalized_crr(sort_y)
                total_NDCG += temp_ndcg(sort_y,threshold=threshold)
            uid += 1
    num_test_user = len(user_list)
    total_NCRR /= num_test_user
    total_NDCG /= num_test_user
    total_recall /= num_test_user
    return total_NDCG, total_NCRR, total_recall

if __name__ == '__main__':
    from config import config
    print(config.test_path)
    print(evaluate_from_file(config.test_path,config.dev_data_path,threshold = config.threshold))


