import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import GCN, GraphUnet, Initializer, norm_g

# random seed
manualSeed = 0
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


class GNet(nn.Module):
    def __init__(self, in_dim, n_classes, args):
        super(GNet, self).__init__()
        self.n_act = getattr(nn, args.act_n)().to(device)
        self.c_act = getattr(nn, args.act_c)().to(device)
        self.s_gcn = GCN(in_dim, args.l_dim, self.n_act, args.drop_n).to(device)
        self.g_unet = GraphUnet(
            args.ks, args.l_dim, args.l_dim, args.l_dim, self.n_act,
            args.drop_n).to(device)
        self.out_l_1 = nn.Linear(3*args.l_dim*(args.l_num+1), args.hidden_dim).to(device)
        self.out_l_2 = nn.Linear(args.hidden_dim, n_classes).to(device)
        self.out_drop = nn.Dropout(p=args.drop_c).to(device)
        Initializer.weights_init(self)

    def forward(self, gs, hs):
        hs = self.embed(gs, hs)
        logits = self.classify(hs)
        return logits
        #return self.metric(logits, labels)

    def embed(self, gs, hs):
        o_hs = []
        for g, h in zip(gs, hs):
            h = self.embed_one(g, h)
            o_hs.append(h)
        hs = torch.stack(o_hs, 0)
        return hs

    def embed_one(self, g, h):
        g = norm_g(g)
        h = self.s_gcn(g, h)
        hs = self.g_unet(g, h)
        h = self.readout(hs)
        return h

    def readout(self, hs):
        h_max = [torch.max(h.to(device), 0)[0] for h in hs]
        h_sum = [torch.sum(h.to(device), 0) for h in hs]
        h_mean = [torch.mean(h.to(device), 0) for h in hs]
        h = torch.cat(h_max + h_sum + h_mean) # # #
        ''' # original
        print([h_max[i].shape for i in range(len(h_max))])
        print([h_sum[i].shape for i in range(len(h_max))])
        print([h_mean[i].shape for i in range(len(h_max))])
        '''
        return h

    def classify(self, h):
        h = self.out_drop(h)
        h = self.out_l_1(h)
        h = self.c_act(h)
        h = self.out_drop(h)
        h = self.out_l_2(h)
        return F.log_softmax(h, dim=1)

    def loss(self, logits, labels):
        loss = F.nll_loss(logits, labels).to(device)
        #_, preds = torch.max(logits, 1)
        #acc = torch.mean((preds == labels).float())
        #return loss, acc
        return loss
