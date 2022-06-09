import gGAN
import torch
import random
import seaborn as sns
import cross_val
import numpy as np
import main_gcn
from torch.autograd import Variable
import matplotlib.pyplot as plt

random.seed(1)
device = torch.device('cpu') #'cuda:0' if torch.cuda.is_available() else

dataset = "RH_ASDNC"
view = 0
args = main_gcn.arg_parse(dataset, view)
G_list = main_gcn.load_data(args)

random.shuffle(G_list)  # shuffle for cbt data
num_nodes = G_list[0]['adj'].shape[0]
folds = cross_val.stratify_splits(G_list[0:int(round(len(G_list) * 0.8))], args)
for_cbt = G_list[int(round(len(G_list) * 0.8)): len(G_list)]
[random.shuffle(folds[i]) for i in range(len(folds))]
for i in range(0, args.cv_number):
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    l1_loss = torch.nn.L1Loss()
    # Initialize generator and discriminator
    discriminator1 = gGAN.Discriminator1()
    learnable_residual = gGAN.LearnableResidual(args.nbr_of_regions)

    discriminator1.to(device)
    learnable_residual.to(device)
    adversarial_loss.to(device)
    l1_loss.to(device)
    trained_model_gen = torch.load('./weightgat3_' + str(i) + 'generator_.model')
    generator = gGAN.Generator()
    generator.to(device)
    #generator = torch.nn.DataParallel(generator)
    generator.load_state_dict(trained_model_gen)

    train_set, validation_set, test_set = cross_val.datasets_splits(folds, args, i)

    if args.evaluation_method == 'model selection':
        train_dataset, val_dataset, threshold_value = cross_val.model_selection_split(train_set, validation_set,
                                                                                      args)

    if args.evaluation_method == 'model assessment':
        train_dataset, val_dataset, threshold_value = cross_val.model_assessment_split(train_set, validation_set,
                                                                                       test_set, args)

    cbt_set_np = np.array([d['adj'] for d in for_cbt])
    CBT = gGAN.netNorm(cbt_set_np, cbt_set_np.shape[0], args.nbr_of_regions)

    target_data = np.reshape(CBT, (1, args.nbr_of_regions, args.nbr_of_regions, 1))
    target_data = torch.from_numpy(target_data)  # convert numpy array to torch tensor
    target_data = target_data.type(torch.FloatTensor)
    train_casted_target = [d.to(device) for d in gGAN.cast_data(target_data, 0)]

    label_0 = []
    label_1 = []
    prev = np.zeros((35, 35))
    label_p = 0
    for batch_idx, data in enumerate(train_dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        label = Variable(data['label'].long()).to(device)
        adj_id = Variable(data['id'].int()).to(device)
        train_casted_source = [d.to(device) for d in gGAN.cast_data(adj, 0)]
        normalized_train_output = gGAN.normalize(args, generator, discriminator1, adversarial_loss, l1_loss,
                                                    train_casted_source, train_casted_target, 1)
        #normalized_cbt_output = gGAN.normalize(args, generator, discriminator1, adversarial_loss, l1_loss,
        #                                       train_casted_target, train_casted_target, 1)

        #residual_train = gGAN.generate_residuals(normalized_train_output, normalized_cbt_output)

        #threshold_value = torch.mean(normalized_train_output[0].detach())
        #adj = torch.where(normalized_train_output[0] > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))

        #print(residual_train[0, np.isclose(residual_train[0].detach().cpu().clone().numpy(), 0)].detach().cpu().clone().numpy())
        if label_p == label:
            print(normalized_train_output - prev)

        if label_p == label:
            label_0.append(np.ndarray.flatten((adj).detach().cpu().clone().numpy()))
        else:
            label_1.append(np.ndarray.flatten((adj).detach().cpu().clone().numpy()))

        prev = normalized_train_output
        label_p = label

    sns.distplot(label_0)

    sns.distplot(label_1)
    # plt.imshow(residual_train[0].detach().cpu().clone().numpy())
    plt.savefig('./residual' + str(i) + '.png')
    plt.clf()