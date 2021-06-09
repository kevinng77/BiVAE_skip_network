# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# https://github.com/PreferredAI/cornac/blob/e29213bd5096a9e001118f5b727e45d9e8457a73/cornac/models/bivaecf/bivae.py

# ============================================================================


import numpy as np
import torch
import torch.nn as nn
import itertools as it

EPS  = 1e-7
ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}


class BiVAE(nn.Module):
    """
    modify the decoder to be skip VAE, trying to solve the prior collapse
    https://arxiv.org/abs/1807.04863

    additional paramenters:
        user_decoder_structure, type : list i.e. [20,20]
        item_decoder_structure, type : list i.e. [20,20]
    """
    def __init__(
        self,
        k,
        user_encoder_structure,
        item_encoder_structure,
        user_decoder_structure,
        item_decoder_structure,
        act_fn,
        likelihood,
        cap_priors,
        feature_dim,
        batch_size,
        hidden_scaler = 1
    ):
        super(BiVAE, self).__init__()

        self.mu_theta = torch.zeros((item_encoder_structure[0], k))  # n_users*k
        self.mu_beta = torch.zeros((user_encoder_structure[0], k))  # n_items*k

        self.theta = torch.randn(item_encoder_structure[0], k) * 0.01
        self.beta = torch.randn(user_encoder_structure[0], k) * 0.01
        torch.nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))

        self.likelihood = likelihood
        self.act_fn = ACT.get(act_fn, None)
        if self.act_fn is None:
            raise ValueError("Supported act_fn: {}".format(ACT.keys()))

        self.cap_priors = cap_priors
        if self.cap_priors.get("user", False):
            self.user_prior_encoder = nn.Linear(feature_dim.get("user"), k)
        if self.cap_priors.get("item", False):
            self.item_prior_encoder = nn.Linear(feature_dim.get("item"), k)

        # User Encoder
        self.user_encoder = nn.Sequential()
        for i in range(len(user_encoder_structure) - 1):
            self.user_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(user_encoder_structure[i], user_encoder_structure[i + 1]),
            )
            self.user_encoder.add_module("act{}".format(i), self.act_fn)
        self.user_mu = nn.Linear(user_encoder_structure[-1], k)  # mu
        self.user_std = nn.Linear(user_encoder_structure[-1], k)

        # Item Encoder
        self.item_encoder = nn.Sequential()
        for i in range(len(item_encoder_structure) - 1):
            self.item_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(item_encoder_structure[i], item_encoder_structure[i + 1]),
            )
            self.item_encoder.add_module("act{}".format(i), self.act_fn)
        self.item_mu = nn.Linear(item_encoder_structure[-1], k)  # mu
        self.item_std = nn.Linear(item_encoder_structure[-1], k)

        self.user_decoder_list = nn.Sequential()
        for i in range(1, len(user_decoder_structure)):
            self.user_decoder_list.add_module(
                "fc_out{}".format(i),
                nn.Linear(user_decoder_structure[i - 1], user_encoder_structure[i])
            )
            self.user_decoder_list.add_module("act_out{}".format(i), self.act_fn)
        self.item_decoder_list = nn.Sequential()
        for i in range(1, len(item_decoder_structure)):
            self.item_decoder_list.add_module(
                "fc_out{}".format(i),
                nn.Linear(item_decoder_structure[i - 1], item_encoder_structure[i])
            )
            self.item_decoder_list.add_module("act_out{}".format(i), self.act_fn)

        # project theta and beta to [num_user, user_decoder_structure[0]]
        self.user_decoder_out = nn.Linear(k, user_decoder_structure[0])
        self.item_decoder_out = nn.Linear(k, item_decoder_structure[0])

        self.hidden_scaler = hidden_scaler

    def to(self, device):
        self.beta = self.beta.to(device=device)
        self.theta = self.theta.to(device=device)
        self.mu_beta = self.mu_beta.to(device=device)
        self.mu_theta = self.mu_theta.to(device=device)

        return super(BiVAE, self).to(device)

    def encode_user_prior(self, x):
        h = self.user_prior_encoder(x)
        return h

    def encode_item_prior(self, x):
        h = self.item_prior_encoder(x)
        return h

    def encode_user(self, x):
        h = self.user_encoder(x)
        return self.user_mu(h), torch.sigmoid(self.user_std(h))

    def encode_item(self, x):
        h = self.item_encoder(x)
        return self.item_mu(h), torch.sigmoid(self.item_std(h))

    def decode_user(self, theta, beta):
        theta_hidden = self.user_decoder_out(theta)
        beta_hidden = self.item_decoder_out(beta)
        # shape theta [num_user, user_decoder_list[0]]
        # beta [num_item, user_decoder_list[0]]
        theta_hidden = self.user_decoder_list(theta_hidden)
        beta_hidden = self.item_decoder_list(beta_hidden)
        # shape theta [num_user, user_decoder_list[-1]]
        # beta [num_item, user_decoder_list[-1]]
        h_hidden = theta_hidden.mm(beta_hidden.t())
        h_hidden = nn.Tanh()(h_hidden)
        h = theta.mm(beta.t())
        return torch.sigmoid(h+h_hidden) * self.hidden_scaler

    def decode_item(self, theta, beta):
        theta_hidden = self.user_decoder_out(theta)
        beta_hidden = self.item_decoder_out(beta)
        theta_hidden = self.user_decoder_list(theta_hidden)
        beta_hidden = self.item_decoder_list(beta_hidden)
        h_hidden = beta_hidden.mm(theta_hidden.t())
        h_hidden = nn.Tanh()(h_hidden)
        h = beta.mm(theta.t())
        return torch.sigmoid(h+h_hidden) * self.hidden_scaler

    def reparameterize(self, mu, std):
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x, user=True, beta=None, theta=None):

        if user:
            mu, std = self.encode_user(x)
            theta = self.reparameterize(mu, std)
            return theta, self.decode_user(theta, beta), mu, std
        else:
            mu, std = self.encode_item(x)
            beta = self.reparameterize(mu, std)
            return beta, self.decode_item(theta, beta), mu, std

    def loss(self, x, x_, mu, mu_prior, std, kl_beta):
        # Likelihood
        ll_choices = {
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -(x - x_) ** 2,
            "pois": x * torch.log(x_ + EPS) - x_,
        }

        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))

        ll = torch.sum(ll, dim=1)

        # KL term
        kld = -0.5 * (1 + 2.0 * torch.log(std) - (mu - mu_prior).pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(kl_beta * kld - ll)

def learn(
    bivae,
    train_set,
    n_epochs,
    batch_size,
    learn_rate,
    beta_kl,
    verbose,
    device=torch.device("cpu"),
    dtype=torch.float32,
    true_threshold = 1
):
    user_params = it.chain(
        bivae.user_encoder.parameters(),
        bivae.user_mu.parameters(),
        bivae.user_std.parameters(),
    )

    item_params = it.chain(
        bivae.item_encoder.parameters(),
        bivae.item_mu.parameters(),
        bivae.item_std.parameters(),
    )

    if bivae.cap_priors.get("user", False):
        user_params = it.chain(user_params, bivae.user_prior_encoder.parameters())
        user_features = train_set.user_feature.features[: train_set.num_users]

    if bivae.cap_priors.get("item", False):
        item_params = it.chain(item_params, bivae.item_prior_encoder.parameters())
        item_features = train_set.item_feature.features[: train_set.num_items]

    u_optimizer = torch.optim.Adam(params=user_params, lr=learn_rate)
    i_optimizer = torch.optim.Adam(params=item_params, lr=learn_rate)

    x = train_set.matrix.copy()
    x.data = np.ones_like(x.data)*(x.data>=true_threshold)  # Binarize data
    tx = x.transpose()

    for epoch in range(1, n_epochs + 1):
        # item side
        i_sum_loss = 0.0
        i_count = 0
        for i_ids in train_set.item_iter(batch_size, shuffle=False):
            i_batch = tx[i_ids, :]
            i_batch = i_batch.A
            i_batch = torch.tensor(i_batch, dtype=dtype, device=device)

            # Reconstructed batch
            beta, i_batch_, i_mu, i_std = bivae(i_batch, user=False, theta=bivae.theta)

            i_mu_prior = 0.0  # zero mean for standard normal prior if not CAP prior
            if bivae.cap_priors.get("item", False):
                i_batch_f = item_features[i_ids]
                i_batch_f = torch.tensor(i_batch_f, dtype=dtype, device=device)
                i_mu_prior = bivae.encode_item_prior(i_batch_f)

            i_loss = bivae.loss(i_batch, i_batch_, i_mu, i_mu_prior, i_std, beta_kl)
            i_optimizer.zero_grad()
            i_loss.backward()
            i_optimizer.step()

            i_sum_loss += i_loss.data.item()
            i_count += len(i_batch)

            beta, _, i_mu, _ = bivae(i_batch, user=False, theta=bivae.theta)

            bivae.beta.data[i_ids] = beta.data
            bivae.mu_beta.data[i_ids] = i_mu.data

        # user side
        u_sum_loss = 0.0
        u_count = 0
        for u_ids in train_set.user_iter(batch_size, shuffle=False):
            u_batch = x[u_ids, :]
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=dtype, device=device)

            # Reconstructed batch
            theta, u_batch_, u_mu, u_std = bivae(u_batch, user=True, beta=bivae.beta)

            u_mu_prior = 0.0  # zero mean for standard normal prior if not CAP prior
            if bivae.cap_priors.get("user", False):
                u_batch_f = user_features[u_ids]
                u_batch_f = torch.tensor(u_batch_f, dtype=dtype, device=device)
                u_mu_prior = bivae.encode_user_prior(u_batch_f)

            u_loss = bivae.loss(u_batch, u_batch_, u_mu, u_mu_prior, u_std, beta_kl)
            u_optimizer.zero_grad()
            u_loss.backward()
            u_optimizer.step()

            u_sum_loss += u_loss.data.item()
            u_count += len(u_batch)

            theta, _, u_mu, _ = bivae(u_batch, user=True, beta=bivae.beta)
            bivae.theta.data[u_ids] = theta.data
            bivae.mu_theta.data[u_ids] = u_mu.data

        if epoch % 20 == 0:
            print(f"epoch: ({epoch}/{n_epochs})\t"
                  f"item loss: {round(i_sum_loss/i_count,3)}\t"
                  f"user loss: {round(u_sum_loss/u_count,3)}")

    # infer mu_beta
    for i_ids in train_set.item_iter(batch_size, shuffle=False):
        i_batch = tx[i_ids, :]
        i_batch = i_batch.A
        i_batch = torch.tensor(i_batch, dtype=dtype, device=device)

        beta, _, i_mu, _ = bivae(i_batch, user=False, theta=bivae.theta)
        bivae.mu_beta.data[i_ids] = i_mu.data

    # infer mu_theta
    for u_ids in train_set.user_iter(batch_size, shuffle=False):
        u_batch = x[u_ids, :]
        u_batch = u_batch.A
        u_batch = torch.tensor(u_batch, dtype=dtype, device=device)

        theta, _, u_mu, _ = bivae(u_batch, user=True, beta=bivae.beta)
        bivae.mu_theta.data[u_ids] = u_mu.data



    return bivae