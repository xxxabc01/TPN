import torch
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_correalation_loss(realtion_feat_1, relation_feat_2, ord_num=2):
    """
    :param matrix1: relation feat from students model
    :param matrix2: relation feat from ensemble model
    :param ord__num: normlize_num
    :return: get the relation loss
    """

    # choose the way to calculate matrix
    matrix1 = torch.matmul(realtion_feat_1, (realtion_feat_1).T)
    # matrix1 = torch.matmul(realtion_feat_1.detach(), (realtion_feat_1.detach()).T)
    # matrix2 = torch.matmul(relation_feat_2, relation_feat_2.T)
    matrix2 = torch.matmul(relation_feat_2.detach(), relation_feat_2.detach().T)


    # get the l2-norm value of the every row of the matrix
    with torch.no_grad():
        matrtix1_row_l2_value = torch.linalg.norm(matrix1, ord_num, dim=1)
        matrtix2_row_l2_value = torch.linalg.norm(matrix2, ord_num, dim=1)

    matrix_row_shape = matrix1.shape[0]
    # calculate the new matrix
    for i in range(matrix_row_shape):
        matrix1[i, :] = matrix1[i, :] / matrtix1_row_l2_value[i]
        matrix2[i, :] = matrix2[i, :] / matrtix2_row_l2_value[i]

    relation_loss = ((matrix1 - matrix2).pow(2) / (matrix_row_shape ** 2)).sum()
    return relation_loss

def calculate_multi_correlation_loss(realation_feat_list_1,relation_feat_list_2,ord_num=2):
    """
    :param realation_feat_list_1:multi_relation_feat from student model
    :param relation_feat_list_2: multi_relation_feat from ensemble model
    :param ord_num: normalize_num
    :return: Get the multi relation loss
    """
    relation_loss = [] # record the loss log from the list
    for i in range(len(realation_feat_list_1)):

        # choose the way to calculate matrix
        matrix1 = torch.matmul(realation_feat_list_1[i], realation_feat_list_1[i].T)
        # matrix1 = torch.matmul(realation_feat_list_1[i].detach(), (realation_feat_list_1[i].detach()).T)
        matrix2 = torch.matmul(relation_feat_list_2[i].detach(), (relation_feat_list_2[i].detach()).T)
        # matrix2 = torch.matmul(relation_feat_list_2[i], relation_feat_list_2[i].detach().T)


        # get the l2-norm value of the every row of the matrix
        with torch.no_grad():
            matrtix1_row_l2_value = torch.linalg.norm(matrix1, ord_num, dim=1)
            matrtix2_row_l2_value = torch.linalg.norm(matrix2, ord_num, dim=1)

        matrix_row_shape = matrix1.shape[0]
        # calculate the new matrix
        for j in range(matrix_row_shape):
            matrix1[j, :] = matrix1[j, :] / matrtix1_row_l2_value[j]
            matrix2[j, :] = matrix2[j, :] / matrtix2_row_l2_value[j]

        middle_relation_loss = ((matrix1 - matrix2).pow(2) / (matrix_row_shape ** 2)).sum()
        relation_loss.append(middle_relation_loss)

    total_realtion_loss = sum(relation_loss)

    return total_realtion_loss

def make_folder(loop_num,seed,M_and_NN_layers,Ensemble_layers,epoch,hidden_dim,lr,lr_phsics,En_f_layer,Re_f_layer,use_uncertainty,use_multi_relation_feat,use_attention,target):
    # get the current time
    import datetime
    current_time = datetime.datetime.now()
    # set the file_time
    # file_name = current_time.strftime("%Y-%m-%d_%H-%M-%S.txt")
    file_title = "loop_num__" + str(loop_num)
    file_time = current_time.strftime("%Y-%m-%d_%H-%M-%S.txt")
    file_name = file_title+file_time
    # folder original path
    folder_path = Path(__file__).parent.parent/'RES'/file_title

    if os.path.exists(Path(__file__).parent.parent/'RES'):
        pass
    else:
        os.mkdir(Path(__file__).parent.parent/'RES')

    # chek whether the dir exists or not
    if os.path.exists(folder_path):
        print("folder already exists：" + str(folder_path))
    else:
        print('Create the new folder: ' + str(folder_path))
        os.mkdir(folder_path)

    # concat the file path
    file_path = os.path.join(folder_path, file_name)
    # write the modle parameter
    with open(file_path, 'w') as file:
        file.write("loop_num: " + str(loop_num) + '\n')

        file.write("Seed: " + str(seed) + '\n')
        file.write("M_and_NN_layers:" + str(M_and_NN_layers)+ '\n')
        file.write("Ensemble_layers:" + str(Ensemble_layers) + '\n')
        file.write("Epoch:" + str(epoch) + '\n')
        file.write("Hidden_dim:" + str(hidden_dim) + '\n')
        file.write("lr:" + str(lr) + '\n')
        file.write("lr_Physics:" + str(lr_phsics) + '\n')
        file.write("En_f_layer:" + str(En_f_layer) + '\n')
        file.write("Re_f_layer:" + str(Re_f_layer) + '\n')
        file.write("use_uncertainty:" + str(use_uncertainty) + '\n')
        file.write("use_multi_relation_feat" + str(use_multi_relation_feat) + '\n')
        file.write("use_attention:" + str(use_attention) + '\n')
        file.write("target" + target + '\n')
    return file_path,file_name

