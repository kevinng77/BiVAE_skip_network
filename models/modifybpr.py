import numpy as np
from cornac.utils.init_utils import zeros, uniform


class MBPR(object):
    def __init__(self,
                 name='MBPR',
                 k=10,
                 max_iter=100,
                 learning_rate=0.001,
                 lambda_reg=0.01,
                 trainable=True,
                 verbose=False,
                 init_params=None,
                 seed=None):
        self.name = name
        self.trainable = trainable
        self.verbose = verbose
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.seed = seed
        self.init_params = {} if init_params is None else init_params
        self.u_factors = self.init_params.get('U', None)
        self.i_factors = self.init_params.get('V', None)
        self.i_biases = self.init_params.get('Bi', None)

    def _init(self,train_set):
        self.train_data = train_set
        n_users, n_items = self.train_data.total_users, self.train_data.total_items

        if self.u_factors is None:
            self.u_factors = (uniform((n_users, self.k)) - 0.5) / self.k
        if self.i_factors is None:
            self.i_factors = (uniform((n_items, self.k)) - 0.5) / self.k
        self.i_biases = zeros(n_items) if self.i_biases is None else self.i_biases

    def _prepare_data(self):
        X = self.train_data.matrix  # csr_matrix
        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(X.indptr)
        user_ids = np.repeat(np.arange(self.train_data.num_users), user_counts).astype(X.indices.dtype)

        return X, user_counts, user_ids

    def _gen_neg(self):
        neg = np.arange(self.train_data.num_items - 1)  # BPR defualt method
        return neg

    def fit(self, train_set, val_set=None):
        self._init(train_set)
        if not self.trainable:
            return self

        X, user_counts, user_ids = self._prepare_data()

        # modify neg ids
        neg_item_ids = np.arange(train_set.num_items, dtype=np.int32)

        #
        rng_pos = np.arange(len(user_ids)-1)
        rng_neg = self._gen_neg()

        for epoch in range(self.max_iter):
            self._fit_sgd(rng_pos, rng_neg,
                          user_ids, X.indices, neg_item_ids, X.indptr,
                          self.u_factors, self.i_factors, self.i_biases)

        return self

    def _fit_sgd(self,pos,neg,user_ids, item_ids, neg_item_ids, indptr,
                                             U, V, B):
        """
        :param pos:
        :param neg:
        :param user_ids: id of user to the corresponding rated item
        :param item_ids: id of positive samples to pick
        :param neg_item_ids: id of negative sample to pick
        :param indptr: CSR matrix index pointer
        :param U: user factor
        :param V: item factor
        :param B: item_bias
        :return:
        """
        num_samples = len(user_ids)
        for uidx in range(self.train_data.num_user):
            # uidx = self.train_data.uid_map[uid]
            beg = indptr[uidx]
            end = indptr[uidx + 1]
            if beg == end:
                continue
            cur_item_idxs = item_ids[beg:end]
            for item_i_idx in cur_item_idxs:
                vec_user, item_i, item_j = U[uidx,:], V[item_i_idx,:],V[j_id,:]

        # compute the score
        score = B[i_id] - B[j_id]


