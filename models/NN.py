import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from Layer.UNN import *
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

se = 25
np.random.seed(se)
torch.manual_seed(se)


# device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# PINN Class
class NN(torch.nn.Module):
    def __init__(self, layers, dim, lb, ub, ensemble_layer_num=2, relation_layer_num=-1, use_uncertainty=False, use_multi_relation_feat=False, device='cpu'):

        super(NN, self).__init__()

        self.lb = lb
        self.ub = ub
        self.layers = layers
        self.device = device

        # ensemble/relation feature layer index and uncertainty option
        self.ensemble_layer_num = ensemble_layer_num
        self.relation_layer_num = relation_layer_num
        self.use_uncertainty = use_uncertainty
        self.use_multi_relation_feat = use_multi_relation_feat

        # model layer layout
        if self.use_uncertainty:
            self.input_layer = BNN_layer(prior_mu=0, prior_sigma=1, in_features=2, out_features=dim)
        else:
            self.input_layer = nn.Linear(2, dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(dim, dim) for i in range(self.layers-2)])
        self.output_layer = nn.Linear(dim, 1)

        # bnn layer
        # self.bnn = BNN_layer(prior_mu=0, prior_sigma=1, in_features=dim, out_features=dim)

        self.mse = nn.MSELoss()
        self.active = nn.Tanh()


    def forward(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        #save the middle feature
        middle_feature = []


        # use bnn to estimate uncertainty or not
        if self.use_uncertainty:
            self.lamda, output = self.input_layer(H)  # lamda represents the uncertainty for feature
            output = self.active(output)
            for i in range(len(self.hidden_layers)):
                output = self.active(self.hidden_layers[i](output))
                middle_feature.append(output)
        else:
            output = self.active(self.input_layer(H))
            for i in range(len(self.hidden_layers)):
                output = self.active(self.hidden_layers[i](output))
                middle_feature.append(output)

        # get the ensemble feature and relation feature
        ensemble_feat = middle_feature[self.ensemble_layer_num]
        # # add uncertainty for ensemble feature
        if self.use_uncertainty:
            ensemble_feat = ensemble_feat * self.lamda


        # get the relation feat form model
        if self.use_multi_relation_feat:
            relation_feat = middle_feature[self.relation_layer_num:]
        else:
            relation_feat = middle_feature[self.relation_layer_num]

        output = self.output_layer(output)

        self.ensemble_feat = ensemble_feat

        self.relation_feat = relation_feat
        return output

    def net_u(self, x, t):
        output = self.forward(torch.cat([x, t], dim=1))
        return output


    def cul_loss(self, X_u, u):
        x_u = X_u[:, 0:1].requires_grad_(True)
        t_u = X_u[:, 1:2].requires_grad_(True)
        self.u_pred = self.net_u(x_u, t_u)

        if self.use_uncertainty:
            self.kl = self.input_layer.bayesian_kl_loss()
            self.loss = self.mse(self.u_pred, u) + 0.1 * self.kl

        else:
            self.loss = self.mse(self.u_pred, u)


        return self.loss