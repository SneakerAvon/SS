import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/PEMS08',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_pems08.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=64,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=170,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/bay',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--method', type=str, default='graph', help='two choices: pure, graph')
parser.add_argument('--fn_t', type=int, default=12, help='filter negatives threshold, 12 means 1 hour')
parser.add_argument('--em_t', type=float, default=0, help='edge masking threshold')
parser.add_argument('--im_t', type=float, default=0, help='input masking threshold')
parser.add_argument('--ts_t', type=float, default=0.5, help='temporal shifting threshold')
parser.add_argument('--ism_t', type=float, default=0, help='input smoothing scale')
parser.add_argument('--ism_e', type=int, default=20,
                    help='input smoothing entries, which means how much entries we keep untouch during scaling in the frequency domain')
parser.add_argument('--tempe', type=float, default=0.1, help='temperature parameter')
parser.add_argument('--lam', type=float, default=0.1, help='loss lambda')
args = parser.parse_args()

device = torch.device(args.device)
adj_mx = util.load_adj(args.adjdata,args.adjtype)
dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
scaler = dataloader['scaler']
supports = [torch.tensor(i).to(device) for i in adj_mx]

print(args)

if args.randomadj:
    adjinit = None
else:
    adjinit = supports[0]

if args.aptonly:
    supports = None

engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                 args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                 adjinit, adj_mx, args.method, args.fn_t, args.em_t, args.im_t, args.ts_t, args.ism_t, args.ism_e,
                 args.tempe, args.lam)

engine.model.load_state_dict(torch.load("./garage/bay_epoch_88_14.49.pth"))

outputs = []
realy = torch.Tensor(dataloader['y_test']).to(device)
realy = realy.transpose(1,3)[:,0,:,:]



for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testx = testx.transpose(1,3)
    with torch.no_grad():
        preds, _ = engine.model(testx)
        preds = preds.transpose(1,3)
    outputs.append(preds.squeeze())

yhat = torch.cat(outputs,dim=0)
yhat = yhat[:realy.size(0),...]

amae = []
amape = []
armse = []
for i in range(12):
    pred = scaler.inverse_transform(yhat[:,:,i])
    real = realy[:,:,i]
    metrics = util.metric(pred,real)
    log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
    amae.append(metrics[0])
    amape.append(metrics[1])
    armse.append(metrics[2])
