import numpy as np
import pandas as pd
import json
import time
import multiprocessing as mp


class MBPR(object):
    def __init__(self,
                 name='MBPR',
                 k=50,
                 max_iter=100,
                 learning_rate=0.01,
                 lambda_reg=0.003,
                 momentum=0.9,
                 verbose=False):
        self.verbose = verbose
        self.k = k
        self.name = f"{name}_k{k}"
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.momentum = momentum  # beta
        self.lambda_reg = lambda_reg
        self.u_factors = None
        self.i_factors = None
        self.i_biases = None
        self.losses = []
        self.acc = 0

    def _init(self, train_set, num_users, num_items):
        """
        input train_set should be pd.DataFrame
        assume the user id and item id is sorted and is continuous.
        :param self.train_set: userid, itemid, y_true(1 or 0)
        :return:
        """
        self.num_users = num_users
        self.num_items = num_items
        self.train_set = train_set.values
        if self.u_factors is None:
            self.u_factors = np.random.random((self.num_users, self.k)) - 0.5
        if self.i_factors is None:
            self.i_factors = np.random.random((self.num_items, self.k)) - 0.5
        self.i_biases = np.zeros(self.num_items) if self.i_biases is None else self.i_biases
        self.grad_u = np.zeros((self.num_users, self.k))
        self.grad_i = np.zeros((self.num_items, self.k))

    def fit(self, train_set, num_users, num_items, num_thread=4, val_set=None):
        self._init(train_set, num_users, num_items)
        time1 = time.time()

        for epoch in range(self.max_iter):
            self._fit_sgd(self.learning_rate,self.lambda_reg,epoch % 5 == 0)

            if epoch % 5 == 0:
                print(f"epoch: {epoch}, acc {self.acc * 100:.2f}% ,"
                      f"criterion: {self.losses[-1]:.6f}, time: {time.time() - time1:.1f}s")
                time1 = time.time()
        return self

    def _fit_sgd(self, learning_rate, lambda_reg,  loss_verbose):
        """
        :param i_factors item factors
        """
        losses = 0
        acc = 0
        num_samples = len(self.train_set)

        for t in range(num_samples):
            uidx = self.train_set[t, 0] - 1
            iidx = self.train_set[t, 1] - 1
            jidx = self.train_set[t, 2] - 1
            u_vector = self.u_factors[uidx, :]
            i_vector = self.i_factors[iidx, :]
            j_vector = self.i_factors[jidx, :]

            a = np.dot(u_vector, i_vector) - np.dot(u_vector, j_vector) + self.i_biases[iidx] - self.i_biases[jidx]
            z = 1 / (1 + np.exp(a))
            if loss_verbose:
                if z < 0.1:
                    acc += 1
                a2 = 1 + np.exp(-a)
                loss = np.log(a2) + learning_rate * (np.linalg.norm(i_vector) + np.linalg.norm(j_vector)) / 2
                losses += loss

            # testing speed
            self.grad_u[uidx, :] = z * (j_vector - i_vector) + lambda_reg * u_vector
            self.grad_i[jidx, :] = (z * u_vector + lambda_reg * j_vector)
            self.grad_i[iidx, :] = (z * -u_vector + lambda_reg * i_vector)

            u_vector -= learning_rate * self.grad_u[uidx, :]
            i_vector -= learning_rate * self.grad_i[iidx, :]
            j_vector -= learning_rate * self.grad_i[jidx, :]
        if loss_verbose:
            self.losses.append(losses/num_samples)
            self.acc = acc/num_samples

    def load(self, checkout_path, model_path=''):
        """
        :param checkout_path: checkout path "../checkout" 
        """
        u_name = checkout_path + f"/model/user_{self.name}.csv"
        i_name = checkout_path + f"/model/item_{self.name}.csv"
        bias_name = checkout_path + f"/model/bias_{self.name}.csv"
        self.u_factors = pd.read_csv(u_name).values
        self.i_factors = pd.read_csv(i_name).values
        self.i_biases = pd.read_csv(bias_name).values.reshape(-1, )

    def save_model(self, checkout_path):
        """
        :param checkout_path: path of checkout folder "./checkout"
        """
        u_name = checkout_path + f"/model/user_{self.name}.csv"
        i_name = checkout_path + f"/model/item_{self.name}.csv"
        bias_name = checkout_path + f"/model/bias_{self.name}.csv"
        pd.DataFrame(self.u_factors).to_csv(u_name, index=False)
        pd.DataFrame(self.i_factors).to_csv(i_name, index=False)
        pd.DataFrame(self.i_biases).to_csv(bias_name, index=False)

    def save_recommend(self, checkout_path):
        y_pred = np.dot(self.u_factors, self.i_factors.T) + self.i_biases.reshape(1, -1)
        recommend_file = checkout_path + '/recommend/' + self.name + '.txt'
        with open(recommend_file, 'w')as fp:
            for uid in range(len(self.u_factors)):
                recom = " ".join([str(x + 1) for x in np.argsort(-y_pred[uid, :])[:50]])
                fp.write(recom + '\n')
        return recommend_file

    def save(self, checkout_path, model_name):
        self.save_model(checkout_path)
        recommend_path = self.save_recommend(checkout_path)
        params = self.get_parameters(checkout_path)
        param_log = checkout_path + '/params/' + model_name
        with open(param_log, "w")as fp:
            json.dump(params, fp, sort_keys=True, indent=1)
        return recommend_path

    def get_parameters(self, checkout_path):
        params = {
            'verbose': self.verbose,
            'k': self.k,
            "name": self.name,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "lambda_reg": self.lambda_reg,
            "model_path": f"{checkout_path}/model",
            "recommend_file_path": f"{checkout_path}/recommend",
            'losses': " ".join([str(round(x, 5)) for x in self.losses]),
        }
        return params
