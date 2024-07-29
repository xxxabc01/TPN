import torch
import torch.nn as nn
import numpy as np
import os
from Layer.UNN import *


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

se = 25
np.random.seed(se)
torch.manual_seed(se)
# device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# PINN Class
class M_NN(torch.nn.Module):
    def __init__(self, layers, dim, lb, ub,ensemble_layer_num=2,relation_layer_num=-1,use_uncertainty=False,use_multi_relation_feat=False, device='cpu'):

        super(M_NN, self).__init__()
        #basic parameter
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
        self.input_layer = nn.Linear(2, dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(dim, dim) for i in range(self.layers-2)])
        self.output_layer = nn.Linear(dim, 1)

        # bnn layer
        self.bnn = BNN_layer(prior_mu=0, prior_sigma=1, in_features=dim, out_features=dim)

        self.mse = nn.MSELoss()
        self.active = nn.Tanh()

        self.rho_max = 0.20
        self.speed_max = 46.64



    def forward(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        # save the middle feature
        middle_feature = []
        output = self.active(self.input_layer(H))

        # use bnn to estimate uncertainty or not(default is false for mnn)
        if self.use_uncertainty:
            self.lamda, output = self.bnn(output)  # lamda represents the uncertainty for feature
            output = self.active(output)
            for i in range(len(self.hidden_layers)):
                output = self.active(self.hidden_layers[i](output))
                middle_feature.append(output)
        else:
            for i in range(len(self.hidden_layers)):
                output = self.active(self.hidden_layers[i](output))
                middle_feature.append(output)

        # get the ensemble feature and relation feature
        ensemble_feat = middle_feature[self.ensemble_layer_num]
        # add uncertainty for ensemble feature
        if self.use_uncertainty:
            ensemble_feat = ensemble_feat*self.lamda



        output = self.output_layer(output)

        self.ensemble_feat = ensemble_feat


        return output

    def net_u(self, x, t):
        output = self.forward(torch.cat([x, t], dim=1))
        return output

    def net_f(self, x, t, target='density'):

        u = self.net_u(x, t)
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]


        if target == 'speed':
            f = self.rho_max * du_dx - 2 * self.rho_max / self.speed_max * u * du_dx - self.rho_max / self.speed_max * du_dt
        elif target == 'density':
            f = self.speed_max * (1 - 2 * u / self.rho_max) * du_dx + du_dt
        else:
            print('Please re-write the estimated traffic states.')
        return f

    def cul_loss(self, X_f, target):

        x_f = X_f[:, 0:1].requires_grad_(True)
        t_f = X_f[:, 1:2].requires_grad_(True)
        self.f_pde = self.net_f(x_f, t_f, target)
        self.loss_pde = torch.mean(torch.square(self.f_pde))


        return self.loss_pde