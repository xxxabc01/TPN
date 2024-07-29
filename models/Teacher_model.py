import torch
import torch.nn as nn
from models.NN import *
from models.M_NN import *
from utils.tools import *


class Teacher_model(nn.Module):
    def __init__(self, M_and_NN_layers, lb, ub, student_ensemble_layer_num, student_relation_layer_num, in_dim, hidden_dim, out_dim,
                 ensemble_middle_layers_count, teacher_ensemble_layer_num=0, teacher_relation_layer_num=-1, atten_option=False,
                 use_multi_relation_feat=False, use_uncertainty=True, device='cpu'):
        super(Teacher_model, self).__init__()

        # set the parameter for ensemble model
        self.model_NN = NN(M_and_NN_layers, hidden_dim, lb, ub, ensemble_layer_num=student_ensemble_layer_num,
                           relation_layer_num=student_relation_layer_num, use_uncertainty=use_uncertainty,
                           use_multi_relation_feat=use_multi_relation_feat, device=device)
        self.model_M_NN = M_NN(M_and_NN_layers, hidden_dim, lb, ub, ensemble_layer_num=student_ensemble_layer_num,
                               relation_layer_num=student_relation_layer_num,
                               use_multi_relation_feat=use_multi_relation_feat, device=device)

        self.teacher_relation_layer_num = teacher_relation_layer_num
        self.atten_option = atten_option
        self.teacher_ensemble_layer_num = teacher_ensemble_layer_num  # layer number to output ensemble feat in the students models
        self.use_relation_feat = False
        self.use_multi_relation_feat = use_multi_relation_feat


        # set hidden layer
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for i in range(ensemble_middle_layers_count - 2)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers_length = len(self.layers)

        self.atten_w = nn.Parameter(torch.FloatTensor(hidden_dim))
        # self.reset_para()

        self.active = nn.Tanh()

        self.mse = nn.MSELoss()

        torch.nn.init.uniform_(self.atten_w)

        # hyper-parameter
        self.rho_max = 0.20
        self.speed_max = 46.64


    def forward(self, X_u_train):
        """
        :param X_u:
        :param u:
        :param X_f:
        :param atten_option: use the attention mechanism or not
        :return: get the middle feature from two student models and output the teacher out
        """

        # NN
        self.NN_pred = self.model_NN(X_u_train)
        self.NN_ensemble_feat = self.model_NN.ensemble_feat
        self.NN_relation_feat = self.model_NN.relation_feat

        # M_NN
        self.M_NN_pred = self.model_M_NN(X_u_train)
        self.M_NN_ensemble_feat = self.model_M_NN.ensemble_feat



        # atten_option
        if self.atten_option:
            # ensemble_feature = torch.stack([self.NN_ensemble_feat, self.M_NN_ensemble_feat], dim=0)
            # ensemble_atten_feature = torch.matmul(ensemble_feature, self.atten_w)
            # ensemble_atten_score = torch.softmax(
            #     torch.cat([ensemble_atten_feature[0], ensemble_atten_feature[1]], dim=1), dim=1)
            # ensemble_model_input = ensemble_atten_score[:, 0].reshape(-1,1) * self.NN_ensemble_feat + ensemble_atten_score[:, 1].reshape(-1,1) * self.M_NN_ensemble_feat
            ensemble_model_input = self.atten_w * self.NN_ensemble_feat + (1 - self.atten_w) * self.M_NN_ensemble_feat
        else:
            # ensemble_model_input = torch.cat([self.NN_ensemble_feat, self.M_NN_ensemble_feat], dim=1)
            ensemble_model_input = self.NN_ensemble_feat + self.M_NN_ensemble_feat

        out = ensemble_model_input
        # save the middle layer feature
        middle_feature = []

        # first layer in model
        out = self.active(self.layers[0](out))

        # hidden layer
        for i in range(1, self.layers_length - 1):
            out = self.active(self.layers[i](out))
            middle_feature.append(out)

        # get the relation feat from the ensemble model
        if self.use_multi_relation_feat:
            ensemble_model_relation_feat = middle_feature[self.teacher_relation_layer_num:]
        else:
            ensemble_model_relation_feat = middle_feature[self.teacher_relation_layer_num]

        self.ensemble_model_relation_feat = ensemble_model_relation_feat

        # finnal layer in the model
        ensemble_model_out = self.layers[-1](out)

        # self.loss = self.mse(ensemble_model_out, u_or_rho_ture)
        return ensemble_model_out

    def net_f(self, x, t, target='density'):
        u = self.forward(torch.cat([x, t], dim=1))
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]


        if target == 'speed':
            f = self.rho_max * du_dx - 2 * self.rho_max / self.speed_max * u * du_dx - self.rho_max / self.speed_max * du_dt
        elif target == 'density':
            f = self.speed_max * (1 - 2 * u / self.rho_max) * du_dx + du_dt
        else:
            print('Please re-write the estimated traffic states.')

        return f

    def calculate_loss(self, X_u_train, u_or_rho_ture):
        """
        :param X_u:
        :param u:
        :param X_f:
        :return: get the teacher_relation_feat,teacher out and loss
        """
        ensemble_model_out = self.forward(X_u_train)
        self.teacher_loss = self.mse(ensemble_model_out, u_or_rho_ture)
        return self.teacher_loss

    def calculate_NN_loss(self, X_u_train, u):
        self.teacher_pred = self.forward(X_u_train)

        self.NN_loss = self.model_NN.cul_loss(X_u_train, u)
        self.NN_pred = self.model_NN.u_pred
        self.teacher_student_mse_loss = self.mse(self.NN_pred, self.teacher_pred)

        if self.use_relation_feat:
            if self.use_multi_relation_feat:
                self.relation_loss = calculate_multi_correlation_loss(self.NN_relation_feat,
                                                                 self.ensemble_model_relation_feat)

                self.NN_loss_all = self.NN_loss + self.teacher_student_mse_loss + self.relation_loss
            else:
                self.relation_loss = calculate_correalation_loss(self.NN_relation_feat,
                                                                      self.ensemble_model_relation_feat)

                self.NN_loss_all = self.NN_loss + self.teacher_student_mse_loss + self.relation_loss
        else:
            self.NN_loss_all = self.NN_loss + self.teacher_student_mse_loss




        return self.NN_loss_all

    def calculate_physics_loss(self, X_f_train, target):
        x_f = X_f_train[:, 0:1].requires_grad_(True)
        t_f = X_f_train[:, 1:2].requires_grad_(True)
        self.teacher_pde = self.net_f(x_f, t_f, target)

        self.physics_loss = self.model_M_NN.cul_loss(X_f_train, target)
        self.student_pde = self.model_M_NN.f_pde
        self.teacher_student_pde_loss = self.mse(self.student_pde, self.teacher_pde)


        self.physics_loss_all = self.physics_loss + self.teacher_student_pde_loss
        # self.physics_loss_all = self.physics_loss

        return self.physics_loss_all

    def reset_para(self):
        """
        :return: reset the atten_layer weight
        """
        torch.nn.init.xavier_uniform_(self.atten_w, gain=1.414)
