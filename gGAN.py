"""Main function of gGAN for the paper: Foreseeing Brain Graph Evolution Over Time
Using Deep Adversarial Network Normalizer
    Details can be found in: (there will be a paper link here)
    (1) the original paper .
    ---------------------------------------------------------------------
    This file contains the implementation of two key steps of our gGAN framework:
        netNorm(v, nbr_of_sub, nbr_of_regions)
                Inputs:
                        v: (n × t x t) matrix stacking the source graphs of all subjects
                            n the total number of subjects
                            t number of regions
                Output:
                        CBT: (t x t) matrix representing the connectional brain template

        gGAN(sourceGraph, nbr_of_regions, nbr_of_folds, nbr_of_epochs, hyper_param1, CBT)
                Inputs:
                        sourceGraph: (n × t x t) matrix stacking the source graphs of all subjects
                                     n the total number of subjects
                                     t number of regions
                        CBT: (t x t) matrix stacking the connectional brain template generated by netNorm

                Output:
                        translatedGraph: (t x t) matrix stacking the graph translated into CBT

    (2) Dependencies: please install the following libraries:
        - matplotlib
        - numpy
        - scikitlearn
        - pytorch
        - pytorch-geometric
        - pytorch-scatter
        - pytorch-sparse
        - scipy

    ---------------------------------------------------------------------
    Copyright 2020 ().
    Please cite the above paper if you use this code.
    All rights reserved.
    """


# If you are using Google Colab please uncomment the three following lines.
# !pip install torch_geometric
# !pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# !pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html


import argparse
import pickle
import os
import pdb
import numpy as np
import math
import itertools
import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from torch_geometric.data import Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv, GCNConv
from torch_geometric.nn import BatchNorm, EdgePooling, TopKPooling, global_add_pool
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats as stats
import seaborn as sns

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('running on GPU')
    # if you are using GPU
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

else:
    device = torch.device("cpu")
    print('running on CPU')

nbr_of_regions = 35


def set_num_regions(num_regions):
    global nbr_of_regions
    nbr_of_regions = num_regions


def netNorm(v, nbr_of_sub):
    nbr_of_feat = int((np.square(nbr_of_regions) - nbr_of_regions) / 2)

    def upper_triangular():
        All_subj = np.zeros((nbr_of_sub, nbr_of_feat))
        for j in range(nbr_of_sub):
            subj_x = v[j, :, :]
            subj_x = np.reshape(subj_x, (nbr_of_regions, nbr_of_regions))
            subj_x = subj_x[np.triu_indices(nbr_of_regions, k=1)]
            subj_x = np.reshape(subj_x, (1, nbr_of_feat))
            All_subj[j, :] = subj_x

        return All_subj

    def distances_inter(All_subj):
        theta = 0
        distance_vector = np.zeros(1)
        distance_vector_final = np.zeros(1)
        x = All_subj
        for i in range(nbr_of_feat):
            ROI_i = x[:, i]
            for j in range(nbr_of_sub):
                subj_j = ROI_i[j:j + 1]

                distance_euclidienne_sub_j_sub_k = 0
                for k in range(nbr_of_sub):
                    if k != j:
                        subj_k = ROI_i[k:k + 1]

                        distance_euclidienne_sub_j_sub_k = distance_euclidienne_sub_j_sub_k + np.square(
                            subj_k - subj_j)
                        theta += 1
                if j == 0:
                    distance_vector = np.sqrt(distance_euclidienne_sub_j_sub_k)
                else:
                    distance_vector = np.concatenate((distance_vector, np.sqrt(distance_euclidienne_sub_j_sub_k)),
                                                     axis=0)

            distance_vector = np.reshape(distance_vector, (nbr_of_sub, 1))
            if i == 0:
                distance_vector_final = distance_vector
            else:
                distance_vector_final = np.concatenate((distance_vector_final, distance_vector), axis=1)

        print(theta)
        return distance_vector_final

    def minimum_distances(distance_vector_final):
        x = distance_vector_final

        for i in range(nbr_of_feat):
            minimum_sub = x[0, i:i + 1]
            minimum_sub = float(minimum_sub)
            general_minimum = 0
            general_minimum = np.array(general_minimum)
            for k in range(1, nbr_of_sub):
                local_sub = x[k:k + 1, i:i + 1]
                local_sub = float(local_sub)
                if local_sub < minimum_sub:
                    general_minimum = k
                    general_minimum = np.array(general_minimum)
                    minimum_sub = local_sub
            if i == 0:
                final_general_minimum = np.array(general_minimum)
            else:
                final_general_minimum = np.vstack((final_general_minimum, general_minimum))

        final_general_minimum = np.transpose(final_general_minimum)

        return final_general_minimum

    def new_tensor(final_general_minimum, All_subj):
        y = All_subj
        x = final_general_minimum
        for i in range(nbr_of_feat):
            optimal_subj = x[:, i:i + 1]
            optimal_subj = np.reshape(optimal_subj, (1))
            optimal_subj = int(optimal_subj)
            if i == 0:
                final_new_tensor = y[optimal_subj: optimal_subj + 1, i:i + 1]
            else:
                final_new_tensor = np.concatenate((final_new_tensor, y[optimal_subj: optimal_subj + 1, i:i + 1]),
                                                  axis=1)

        return final_new_tensor

    def make_sym_matrix(nbr_of_regions, feature_vector):
        my_matrix = np.zeros([nbr_of_regions, nbr_of_regions], dtype=np.double)

        my_matrix[np.triu_indices(nbr_of_regions, k=1)] = feature_vector
        my_matrix = my_matrix + my_matrix.T
        my_matrix[np.diag_indices(nbr_of_regions)] = 0

        return my_matrix

    def re_make_tensor(final_new_tensor, nbr_of_regions):
        x = final_new_tensor
        # x = np.reshape(x, (nbr_of_views, nbr_of_feat))

        x = make_sym_matrix(nbr_of_regions, x)
        x = np.reshape(x, (1, nbr_of_regions, nbr_of_regions))

        return x

    Upp_trig = upper_triangular()
    Dis_int = distances_inter(Upp_trig)
    Min_dis = minimum_distances(Dis_int)
    New_ten = new_tensor(Min_dis, Upp_trig)
    Re_ten = re_make_tensor(New_ten, nbr_of_regions)
    Re_ten = np.reshape(Re_ten, (nbr_of_regions, nbr_of_regions))
    np.fill_diagonal(Re_ten, 0)
    network = np.array(Re_ten)
    return network

def cast_data(array_of_tensors, version):
    version1 = torch.tensor(version, dtype=torch.int)

    N_ROI = array_of_tensors[0].shape[0]
    CHANNELS = 1
    dataset = []
    edge_index = torch.zeros(2, N_ROI * N_ROI)
    edge_attr = torch.zeros(N_ROI * N_ROI, CHANNELS)
    x = torch.zeros((N_ROI, N_ROI))  # 35 x 35
    y = torch.zeros((1,))

    counter = 0
    for i in range(N_ROI):
        for j in range(N_ROI):
            edge_index[:, counter] = torch.tensor([i, j])
            counter += 1
    for mat in array_of_tensors:  # 1,35,35,4

        if version1 == 0:
            edge_attr = mat.view((nbr_of_regions * nbr_of_regions), 1)
            x = mat.view(nbr_of_regions, nbr_of_regions)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            x = torch.tensor(x, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            dataset.append(data)

        elif version1 == 1:
            edge_attr = torch.randn(N_ROI * N_ROI, CHANNELS)
            x = torch.randn(N_ROI, N_ROI)  # 35 x 35
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            x = torch.tensor(x, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            dataset.append(data)

    return dataset

# ------------------------------------------------------------

def plotting_loss(losses_generator, losses_discriminator, epoch):
    plt.figure(1)
    plt.plot(epoch, losses_generator, 'r-')
    plt.plot(epoch, losses_discriminator, 'b-')
    plt.legend(['G Loss', 'D Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./plot/loss' + str(epoch) + '.png')

# -------------------------------------------------------------


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        nn = Sequential(Linear(1, (nbr_of_regions * nbr_of_regions)), ReLU())
        self.conv1 = NNConv(nbr_of_regions, nbr_of_regions, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(nbr_of_regions, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, nbr_of_regions), ReLU())
        self.conv2 = NNConv(nbr_of_regions, 1, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, nbr_of_regions), ReLU())
        self.conv3 = NNConv(1, nbr_of_regions, nn, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(nbr_of_regions, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)

        x3 = torch.cat([F.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr))), x1], dim=1)
        x4 = x3[:, 0:nbr_of_regions]
        x5 = x3[:, nbr_of_regions:2 * nbr_of_regions]

        x6 = (x4 + x5) / 2
        return (x6 + torch.transpose(x6, 0, 1)) / 2


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        nn = Sequential(Linear(1, (nbr_of_regions * nbr_of_regions)), ReLU())
        self.conv1 = NNConv(nbr_of_regions, nbr_of_regions, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(nbr_of_regions, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, nbr_of_regions), ReLU())
        self.conv2 = NNConv(nbr_of_regions, 1, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv22(self.conv2(x, edge_index, edge_attr)))

        return F.sigmoid((x + torch.transpose(x, 0, 1)) / 2)

# ----------------------------------------
#                Training
# ----------------------------------------


def register(args, generator, discriminator1, adversarial_loss, l1_loss, train_casted_source, train_casted_target,
              type):

    # Train Generator
    with torch.autograd.set_detect_anomaly(True):
        registered_outputs = []

        for data_A in train_casted_source:
            generators_output_ = generator(data_A).to(device)  # 35 x35
            if type == 1:
                registered_outputs.append(generators_output_.detach())
            else:
                registered_outputs.append(generators_output_)
                generators_output = generators_output_.view(1, args.nbr_of_regions, args.nbr_of_regions, 1).type(
                    torch.FloatTensor)
            if type == 0:
                generators_output_casted = [d.to(device) for d in cast_data(generators_output, 0)]
                for (data_discriminator) in generators_output_casted:
                    discriminator_output_of_gen = discriminator1(data_discriminator).to(device)
                    g_loss_adversarial = adversarial_loss(discriminator_output_of_gen,
                                                          torch.ones_like(discriminator_output_of_gen).to(device))

                    g_loss_pix2pix = l1_loss(generators_output_,
                                             train_casted_target[0].edge_attr.view(args.nbr_of_regions,
                                                                                   args.nbr_of_regions))

                    g_loss = g_loss_adversarial + (args.hyper_param1 * g_loss_pix2pix)
                    loss_generator = g_loss

                    discriminator_output_for_real_loss = discriminator1(train_casted_target[0]).to(device)

                    real_loss = adversarial_loss(discriminator_output_for_real_loss,
                                                 (torch.ones_like(discriminator_output_for_real_loss,
                                                                  requires_grad=False).to(device)))
                    fake_loss = adversarial_loss(discriminator_output_of_gen.detach(),
                                                 torch.zeros_like(discriminator_output_of_gen).to(device))

                    d_loss = (real_loss + fake_loss) / 2
                    loss_discriminator = d_loss

    if type == 0:
        return loss_generator, loss_discriminator, torch.stack(registered_outputs)
    else:
        return torch.stack(registered_outputs)
