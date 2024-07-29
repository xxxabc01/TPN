import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io

import time

import pandas as pd
import os
import argparse
#import model
from models.NN import *
from models.M_NN import *
from models.Teacher_model import *
from utils.tools import *


se = 25
np.random.seed(se)
torch.manual_seed(se)

"""
    Parameter Setting
"""
parser = argparse.ArgumentParser(description="Parameter.")
parser.add_argument('--N_u', type=int, default=144, help='The number of sample training for net_u')
parser.add_argument('--N_f', type=int, default=12000, help='The number of sample training for net_f')
parser.add_argument('--seed', type=int, default=25, help='Random seed.')
parser.add_argument('--M_and_NN_layers', type=int, default=3, help='The layer numbers of Physic-Informed and Data-Driven model.')
parser.add_argument('--Ensemble_layers', type=int, default=3, help='The layer numbers of Ensemble teacher model.')
parser.add_argument('--epoch', type=int, default=20000, help='The epoch numbers for training')
parser.add_argument('--hidden_dim', type=int, default=20, help='The hidden feature dim of hidden neural layers')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for Ensemble teacher model training')
parser.add_argument('--lr_NN', type=float, default=0.01, help='Learning rate for Data-Driven model training')
parser.add_argument('--lr_physics', type=float, default=0.0005, help='Learning rate for Physic-Informed model training')
parser.add_argument('--student_ensemble_layer_num', type=int, default=0,help='The index of hidden layer in Physic-Informed and Data-Driven model to ensemble feature')
parser.add_argument('--student_relation_layer_num', type=int, default=0,help='The index of hidden layer in Data-Driven model to output feature to calculate relation loss')
parser.add_argument('--use_uncertainty', type=bool, default=True, help='Use uncertainty for Data-Driven model or not')
parser.add_argument('--use_attention', type=bool, default=False,help='Use attention mechanism or not when ensemble feature')
# parser.add_argument('--batch_size', type=int, default=64, help='The size of one batch data')
parser.add_argument('--use_multi_relation_feat', type=bool, default=False,help='Use middle feat to calculate relation matrix or not')
parser.add_argument('--predict_target', type=str, default='speed',help='traffic states, select from [density, speed]')
parser.add_argument('--device_num',type=int,default=3,help='GPU device number')
parser.add_argument('--loop_num',type=int,default=8,help='Loop number')


args = parser.parse_args()
print(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


"""
load the input data
"""
############################### Input Data #################################
from  utils.load_ngsim_data import *

# load the data
load_model = ngsim_data_loader(Loop_number=args.loop_num)
load_model.load_data()

X_star = load_model.X_star

u_star = load_model.u_star

rho_star = load_model.rho_star

lb = load_model.lb
ub = load_model.ub
lb = torch.from_numpy(lb).to(dtype=torch.float32)
ub = torch.from_numpy(ub).to(dtype=torch.float32)

############################### Training Data #################################


X_u_train = load_model.X_u_train


u_train = load_model.u_train


rho_train = load_model.rho_train

X_f_train = load_model.X_f_train
#
X_u_train = torch.from_numpy(X_u_train).to(dtype=torch.float32)
u_train = torch.from_numpy(u_train).to(dtype=torch.float32)
u_star = torch.from_numpy(u_star).to(dtype=torch.float32)

rho_train = torch.from_numpy(rho_train).to(dtype=torch.float32)
X_f_train = torch.from_numpy(X_f_train).to(dtype=torch.float32)
rho_star = torch.from_numpy(rho_star).to(dtype=torch.float32)
X_star = torch.from_numpy(X_star).to(dtype=torch.float32)

############################### Training Data #################################

#GPU

device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

lb = lb.to(device)
ub = ub.to(device)

X_u_train = X_u_train.to(device)
rho_train = rho_train.to(device)
u_train = u_train.to(device)
X_f_train = X_f_train.to(device)
X_star = X_star.to(device)
rho_star = rho_star.to(device)
u_star = u_star.to(device)


"""
save the result 
"""
file_path,file_name = make_folder(args.loop_num,args.seed,args.M_and_NN_layers,args.Ensemble_layers,args.epoch,args.hidden_dim,
                args.lr,args.lr_physics,args.student_ensemble_layer_num,args.student_relation_layer_num,args.use_uncertainty,args.use_multi_relation_feat,args.use_attention,target=args.predict_target)





############################### Training Data #################################

"""
load the model 
"""
############################### Load ensemble model #################################
M_and_NN_layers = args.M_and_NN_layers
M_and_NN_dim = args.hidden_dim
lr = args.lr
lr_NN = args.lr_NN
lr_physics = args.lr_physics
epoch = args.epoch
ensemble_model_input_dim = args.hidden_dim
ensemble_model_hidden_dim = args.hidden_dim
ensemble_model_out_dim = 1 #out dim for regression task
ensemble_middle_layers_count = args.Ensemble_layers

model_ensemble = Teacher_model(M_and_NN_layers, lb, ub, args.student_ensemble_layer_num, args.student_relation_layer_num, ensemble_model_input_dim,
                               ensemble_model_hidden_dim, ensemble_model_out_dim,
                               ensemble_middle_layers_count=ensemble_middle_layers_count,
                               teacher_ensemble_layer_num=args.student_ensemble_layer_num,
                               teacher_relation_layer_num=args.student_relation_layer_num, atten_option=args.use_attention,
                               use_multi_relation_feat=args.use_multi_relation_feat, use_uncertainty=args.use_uncertainty,
                               device=device)

model_ensemble = model_ensemble.to(device)
optimizer_model_ensemble = torch.optim.Adam(model_ensemble.parameters(), lr=lr)
optimizer_NN = torch.optim.Adam(model_ensemble.model_NN.parameters(), lr=lr_NN)
optimizer_M_NN = torch.optim.Adam(model_ensemble.model_M_NN.parameters(), lr=lr_physics)
############################### Load ensemble model #################################

time1 = time.time()

teacher_test = 1e10
teacher_test_epoch = 0

best_train_error = 1e10
best_train_epoch  = 0



for i in range(1, epoch+1):
    model_ensemble.train()
    if args.predict_target == 'speed':
        optimizer_model_ensemble.zero_grad()
        teacher_loss = model_ensemble.calculate_loss(X_u_train, u_train)
        teacher_loss.backward()
        optimizer_model_ensemble.step()

        optimizer_NN.zero_grad()
        NN_loss = model_ensemble.calculate_NN_loss(X_u_train, u_train)
        NN_loss.backward()
        optimizer_NN.step()

        optimizer_M_NN.zero_grad()
        physics_loss = model_ensemble.calculate_physics_loss(X_f_train, args.predict_target)
        physics_loss.backward()
        optimizer_M_NN.step()


        u_pred_PINN_train = model_ensemble(X_u_train)
        error_u_PINN_trian = torch.linalg.norm(u_train - u_pred_PINN_train, 2) / torch.linalg.norm(u_train, 2)


        # Teacher
        u_pred_teacher = model_ensemble(X_star)
        error_u_teacher = torch.linalg.norm(u_star - u_pred_teacher, 2) / torch.linalg.norm(u_star, 2)


        # REVISESD
        if error_u_PINN_trian<best_train_error:
            best_train_error = error_u_PINN_trian
            best_train_epoch = i
            if error_u_teacher < teacher_test:
                teacher_test = error_u_teacher




        #
        print("Epoch:{0:05d} | Ensemble_loss:{1:.15f} | NN_loss:{2:.5f} | M_NN_loss:{3:.15f} ".format(i, teacher_loss, NN_loss, physics_loss))

    elif args.predict_target == 'density':
        optimizer_model_ensemble.zero_grad()
        teacher_loss = model_ensemble.calculate_loss(X_u_train, rho_train)
        teacher_loss.backward()
        optimizer_model_ensemble.step()

        optimizer_NN.zero_grad()
        NN_loss = model_ensemble.calculate_NN_loss(X_u_train, rho_train)
        NN_loss.backward()
        optimizer_NN.step()

        optimizer_M_NN.zero_grad()
        physics_loss = model_ensemble.calculate_physics_loss(X_f_train, args.predict_target)
        physics_loss.backward()
        optimizer_M_NN.step()


        #add

        rho_pred_PINN_train = model_ensemble(X_u_train)
        error_rho_PINN_trian = torch.linalg.norm(rho_train - rho_pred_PINN_train, 2) / torch.linalg.norm(rho_train, 2)

        # Teacher
        rho_pred_teacher = model_ensemble(X_star)
        error_rho_teacher = torch.linalg.norm(rho_star - rho_pred_teacher, 2) / torch.linalg.norm(rho_star, 2)

        if error_rho_PINN_trian<best_train_error:
            best_train_error = error_rho_PINN_trian
            best_train_epoch = i
            if error_rho_teacher < teacher_test:
                teacher_test = error_rho_teacher
                teacher_test_epoch = i




        print("Epoch:{0:05d} | Ensemble_loss:{1:.8f} | NN_loss:{2:.8f} | M_NN_loss:{3:.15f} ".format(i, teacher_loss, NN_loss, physics_loss))

    else:
        print('Please re-write the estimated traffic states.')



time2 = time.time()
print("Total time:{0:.4f}".format(time2-time1))

"""
Prediction result
"""

print('Teacher test result:{0:.4f}'.format(teacher_test))



