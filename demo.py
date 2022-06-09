import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io
import argparse
import main_trainer

def arg_parse(dataset, model):
    """
    args definition method
    """
    parser = argparse.ArgumentParser(description='Graph Classification')
    parser.add_argument('--dataset', type=str, default=dataset,
                        help='Dataset')
    parser.add_argument('--model', type=str, default=model,
                        help='Classifier model')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Training Epochs')
    parser.add_argument('--nbr_of_regions', type=int, default=35,
                        help='Number of regions')
    parser.add_argument('--hyper_param1', type=int, default=200,
                        help='Hyper parameter for L1 loss to adversarial loss')
    parser.add_argument('--cv_number', type=int, default=5,
                        help='number of validation folds')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate of the GNN classifier')
    parser.add_argument('--lr_G', type=float, default=0.0001,
                        help='Initial learning rate of the registrator')
    parser.add_argument('--lr_D', type=float, default=0.001,
                        help='Initial learning rate of the disriminator')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--threshold', dest='threshold', default='mean',
                        help='threshold the graph adjacency matrix. Possible values: no_threshold, median, mean')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=512,
                        help='Hidden dimension (diffpool, gunet)')
    parser.add_argument('--output-dim', dest='output_dim', type=int, default=72,
                        help='Output dimension (diffpool)')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=2,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                        help='Number of graph convolution layers before each pooling (diffpool)')
    parser.add_argument('--assign-ratio', dest='assign_ratio', type=float, default=0.1,
                        help='ratio of number of nodes in consecutive layers (diffpool)')
    parser.add_argument('--num-pool', dest='num_pool', type=int, default=1,
                        help='number of pooling layers (diffpool)')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used (diffpool)')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used (diffpool)')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True (diffpool)')
    parser.add_argument('--clip', dest='clip', type=float, default=2.0,
                        help='Gradient clipping (diffpool)')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units (gat, gcn)')
    parser.add_argument('--nb_heads', type=int, default=16,
                        help='Number of head attentions (gat)')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha for the leaky_relu (gat)')
    parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree (gunet)')
    parser.add_argument('-l_num', type=int, default=3, help='layer num (gunet)')
    parser.add_argument('-l_dim', type=int, default=144, help='layer dim (gunet)')
    parser.add_argument('-drop_n', type=float, default=0.0, help='drop net (gunet)')
    parser.add_argument('-drop_c', type=float, default=0.0, help='drop output (gunet)')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act (gunet)')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act (gunet)')
    parser.add_argument('-ks', nargs='+', type=float, default=[0.9, 0.8, 0.7], help='(gunet)')
    parser.add_argument('-acc_file', type=str, default='re', help='acc file (gunet)')

    return parser.parse_args()

# possible values: RH_ASDNC, LH_ASDNC, RH_ADLMCI, LH_ADLMCI, simulated
dataset = "simulated"

# possible values: DIFFPOOL, GCN, GAT, GUNET
model = ["GUNET", "GAT", "DIFFPOOL", "GCN"]

for m in model:
    args = arg_parse(dataset, m)
    print("Main : ", args)
    test_accs = main_trainer.test_scores(args)


