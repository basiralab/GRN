import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle
import scipy
from torch.autograd import Variable
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

import cross_val
import models_diffpool as model_diffpool
from models_gcn import GCN
from models_gunet import GNet
from models_gat import GAT
import gGAN

import time
import random

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import mlab
from os import path

from utils.plot import plot_matrix

# random seed
manualSeed = 0

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if torch.cuda.is_available():
    device = torch.device('cuda')
    # if you are using GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    device = torch.device("cpu")


def evaluate(dataset, CBT, model, generator, discriminator, args, fold, epoch):
    """
    Parameters
    ----------
    dataset : dataloader (dataloader for the validation/test dataset).
    model : nn model (diffpool, gat, gunet or gcn).
    args : arguments
    threshold_value : float (threshold for adjacency matrices).

    Description
    ----------
    This methods performs the evaluation of the model on test/validation dataset

    Returns
    -------
    test accuracy.
    """
    model.eval()
    labels = []
    preds = []

    generator.eval()
    discriminator.eval()

    target_data = np.reshape(CBT, (1, args.nbr_of_regions, args.nbr_of_regions, 1))
    target_data = torch.from_numpy(target_data)  # convert numpy array to torch tensor
    target_data = target_data.type(torch.FloatTensor)
    train_casted_target = [d.to(device) for d in gGAN.cast_data(target_data, 0)]

    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    l1_loss = torch.nn.L1Loss()

    adversarial_loss.to(device)
    l1_loss.to(device)

    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        test_casted_source = [d.to(device) for d in gGAN.cast_data(adj, 0)]
        registered_test_output = gGAN.register(args, generator, discriminator, adversarial_loss, l1_loss,
                                                test_casted_source, train_casted_target, 1)

        if epoch == args.num_epochs - 1:
            plot_matrix(adj[0], args.model, batch_idx, fold)
            plot_matrix(registered_test_output[0], "registered_" + args.model, batch_idx, fold)

        adj = registered_test_output[0]
        labels.append(data['label'].long().numpy())

        batch_num_nodes = np.array([adj.shape[1]])

        assign_input = np.identity(adj.shape[1])
        assign_input = Variable(torch.from_numpy(assign_input).float(), requires_grad=False)

        if args.threshold == "median":
            threshold_value = torch.median(adj.detach())
            adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))
        if args.threshold == "mean":
            threshold_value = torch.mean(adj.detach())
            adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))

        if args.model == "DIFFPOOL":
            assign_input = torch.unsqueeze(assign_input, 0)
            ypred = model(assign_input, adj, batch_num_nodes, assign_x=assign_input)
        elif args.model == "GCN":
            adj = torch.squeeze(adj)
            ypred = model(assign_input, adj)
        elif args.model == "GUNET":
            adj = torch.squeeze(adj)
            ypred = model([adj], [assign_input])
        elif args.model == "GAT":
            ypred = model(assign_input, adj)

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}

    print("Test accuracy:", result['acc'])
    return result['acc']


def minmax_sc(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x


def train(args, train_dataset, val_dataset, for_cbt, model, fold):
    """
    Parameters
    ----------
    args : arguments
    train_dataset : dataloader (dataloader for the train dataset).
    val_dataset : dataloader (dataloader for the validation/test dataset).
    model : nn model (diffpool, gat, gunet, gcn).

    Description
    ----------
    This methods performs the training of the model on train dataset and calls evaluate() method for evaluation.

    Returns
    -------
    test accuracy.
    """
    # -------------------------- #
    #         Registrator
    # -------------------------- #

    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    l1_loss = torch.nn.L1Loss()
    # loss coefficient
    lam = 40

    # set number of regions for gGAN functions
    gGAN.set_num_regions(args.nbr_of_regions)

    # acquire CBT using netNorm
    cbt_set_np = np.array([d['adj'] for d in for_cbt])
    CBT = gGAN.netNorm(cbt_set_np, cbt_set_np.shape[0])

    # target data (CBT) to torch Float Tensor format
    target_data = np.reshape(CBT, (1, args.nbr_of_regions, args.nbr_of_regions, 1))
    target_data = torch.from_numpy(target_data)  # convert numpy array to torch tensor
    target_data = target_data.type(torch.FloatTensor)

    # Initialize generator and discriminator

    generator = gGAN.Generator()
    discriminator = gGAN.Discriminator()

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)
    l1_loss.to(device)

    # Optimizers
    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=args.lr_G, betas=(0.5, 0.999)) # 0.0001
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_D, betas=(0.5, 0.999)) # 0.001

    train_casted_target = [d.to(device) for d in gGAN.cast_data(target_data, 0)]

    # --------------------------- #
    #     classifier
    # --------------------------- #

    model.to(device)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    test_accs = []
    total_losses = []
    plot_loss_g = np.empty((args.num_epochs))
    plot_loss_d = np.empty((args.num_epochs))

    for epoch in range(args.num_epochs):
        print("Epoch ", epoch)

        model.train()
        generator.train()
        discriminator.train()

        total_time = 0
        avg_loss = 0.0

        preds = []
        labels = []
        losses_discriminator = []
        losses_generator = []

        for batch_idx, data in enumerate(train_dataset):
            begin_time = time.time()

            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            train_casted_source = [d.to(device) for d in gGAN.cast_data(adj, 0)]

            loss_generator, loss_discriminator, registered_train_output = gGAN.register(args, generator,
                                                                                         discriminator,
                                                                                         adversarial_loss, l1_loss,
                                                                                         train_casted_source,
                                                                                         train_casted_target, 0)

            adj = registered_train_output[0]
            label = Variable(data['label'].long()).to(device)

            batch_num_nodes = np.array([adj.shape[1]])

            assign_input = np.identity(adj.shape[1])
            assign_input = Variable(torch.from_numpy(assign_input).float(), requires_grad=False).to(device)

            if args.threshold == "median":
                threshold_value = torch.median(adj.detach())
                adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))
            if args.threshold == "mean":
                threshold_value = torch.mean(adj.detach())
                adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))

            if args.model == "DIFFPOOL":
                assign_input = torch.unsqueeze(assign_input, 0)
                ypred = model(assign_input, adj, batch_num_nodes, assign_x=assign_input)
            elif args.model == "GCN":
                adj = torch.squeeze(adj)
                ypred = model(assign_input, adj)
            elif args.model == "GUNET":
                adj = torch.squeeze(adj)
                ypred = model([adj], [assign_input])
            elif args.model == "GAT":
                ypred = model(assign_input, adj)

            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())

            loss = model.loss(ypred, label)
            avg_loss += loss

            losses_generator.append(loss_generator)
            losses_discriminator.append(loss_discriminator)

            optimizer.zero_grad()
            optimizer_G.zero_grad()

            total_loss = lam * loss + loss_generator
            total_loss.backward(retain_graph=True)
            optimizer_G.step()
            optimizer.step()

            optimizer_D.zero_grad()
            loss_discriminator.backward(retain_graph=True)
            optimizer_D.step()

            elapsed = time.time() - begin_time
            total_time += elapsed

        avg_g = torch.mean(torch.stack(losses_generator))
        avg_d = torch.mean(torch.stack(losses_discriminator))
        torch.save(generator.state_dict(), "./weights/" + args.model + "_" + args.dataset + "_" + str(fold) + "generator.model")
        torch.save(discriminator.state_dict(), "./weights/" + args.model + "_" + args.dataset + "_" + str(fold) + "discriminator.model")
        torch.save(model.state_dict(), "./weights/" + args.model + "_" + args.dataset + "_" + str(fold) + ".model")
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f] [total loss: %f]"
            % (epoch, args.num_epochs, avg_d, avg_g, avg_loss))

        plot_loss_g[epoch] = avg_g
        plot_loss_d[epoch] = avg_d
        count = avg_g

        elapsed = time.time() - begin_time
        total_time += elapsed

        total_losses.append(avg_loss.detach().numpy())
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        print("Train accuracy : ", np.mean(preds == labels))
        test_acc = evaluate(val_dataset, CBT, model, generator, discriminator, args, fold, epoch)

        print('Avg classification loss: ', avg_loss.detach().numpy() / len(train_dataset), '; epoch time: ', total_time)

    return test_acc, total_losses


def load_data():
    """
    Description
    ----------
    This methods loads the adjacency matrices of brain graphs

    Returns
    -------
    List of dictionaries{adj, label, id}
    """
    graphs_0 = np.load("./data/multivariate_simulation_data_0.npy")
    graphs_1 = np.load("./data/multivariate_simulation_data_1.npy")

    graphs = np.concatenate((graphs_0, graphs_1), axis=0)

    labels = np.zeros((len(graphs)))
    labels[len(graphs_0):] = 1

    # Create List of Dictionaries
    G_list = []
    for i in range(len(labels)):
        G_element = {"adj": graphs[i], "label": labels[i], "id": i, }
        G_list.append(G_element)
    return G_list


def train_and_evaluate(args):
    """
    Parameters
    ----------
    args : Arguments
    Description
    ----------
    Initiates the model and performs train/test or train/validation splits and calls train() to execute training and evaluation.
    Returns
    -------
    test_accs : test accuracies (list)

    """
    test_accs = []
    # load data split it into for_cbt and folds
    G_list = load_data()
    random.shuffle(G_list)  # shuffle for cbt data
    num_nodes = G_list[0]['adj'].shape[0]

    folds = cross_val.stratify_splits(G_list[0:int(round(len(G_list) * 0.8))], args) # [0:int(round(len(G_list) * 0.8))]
    for_cbt = G_list[int(round(len(G_list) * 0.8)): len(G_list)]
    [random.shuffle(folds[i]) for i in range(len(folds))]
    for i in range(0, args.cv_number):
        train_set, validation_set, test_set = cross_val.datasets_splits(folds, args, i)

        train_dataset, val_dataset = cross_val.model_assessment_split(train_set, validation_set,
                                                                                           test_set, args)

        assign_input = num_nodes
        input_dim = num_nodes
        print("CV : ", i)
        if args.model == "DIFFPOOL":
            model = model_diffpool.SoftPoolingGcnEncoder(
            num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
            assign_input_dim=assign_input)

        elif args.model == "GAT":
            model = GAT(nfeat=num_nodes,
                            nhid=args.hidden,
                            nclass=args.num_classes,
                            dropout=args.dropout,
                            nheads=args.nb_heads,
                            alpha=args.alpha)

        elif args.model == "GUNET":
            model = GNet(num_nodes, args.num_classes, args)

        elif args.model == "GCN":
            model = GCN(nfeat=num_nodes,
                            nhid=args.hidden,
                            nclass=args.num_classes,
                            dropout=args.dropout)

        test_acc, total_losses = train(args, train_dataset, val_dataset, for_cbt, model, i)
        plt.figure(i + 5)
        indexes = np.arange(len(total_losses))
        plt.plot(indexes, total_losses, 'r-')
        plt.legend('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        title = str(i) + "_" + args.model
        plt.savefig(title + ".png")
        # show
        # plt.show()
        plt.clf()
        test_accs.append(test_acc)
    cv_number = len(test_accs)
    acc_std = np.std(test_accs)
    test_accs.append(np.mean(test_accs))
    test_accs = [round(test_accs[i] * 100, 2) for i in range(len(test_accs))]
    x = np.arange(len(test_accs))
    x_labels = ["fold " + str(i + 1) for i in range(cv_number)] + ["mean"]
    stds = [0] * cv_number + [acc_std * 100]
    plt.figure(cv_number)
    ax = plt.subplot(111)
    up = max(test_accs) * .10
    ax.bar(x, test_accs, yerr=stds, align='center', color=(0.5, 0.1, 0.5, 0.6))
    for xi, yi, l in zip(*[x, test_accs, list(map(str, test_accs))]):
        ax.text(xi - len(l) * .05, yi - up, l)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', which='major', labelsize=12)
    plt.title('accs using training set level threshold')
    plt.savefig("./plots/acc_" + args.model + ".png")
    plt.clf()
    return test_accs


def test_scores(args):
    test_accs = train_and_evaluate(args)
    print("test accuracies ", test_accs)

    return test_accs



