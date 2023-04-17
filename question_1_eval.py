import torch
import argparse
import os
from utils.metrics import *
from networks.models import Adaline, BP


parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden', type=int, help='number of neurons in hidden layer')
parser.add_argument('--n_layer', type=int, default=1, help='number of layers in network')
parser.add_argument('--net', type=str, default='bp', help='type of net: [bp, adaline]')
parser.add_argument('--activate', type=str, default='sigmoid', help='type of activation layer: [sigmoid, relu]')
parser.add_argument('--weights', type=str, help='path to weights of pre-trained model')
args = parser.parse_args()

with open('datasets/WindFarm_Data/Q1-wind farm data.txt', 'r') as f:
    lines = f.readlines()
test_X = [[float(line.split()[1]), float(line.split()[2])] for line in lines[124:155]]
test_y = [[float(line.split()[3])] for line in lines[124:155]]
test_X = torch.tensor(test_X)
test_y = torch.tensor(test_y)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_X = test_X.to(DEVICE)
test_y = test_y.to(DEVICE)

net = args.net
activate = args.activate
n_layer = args.n_layer
if net == 'bp':
    model = BP(2, args.n_hidden, 1, n_layer, activate).to(DEVICE)
elif net == 'adaline':
    model = Adaline(2, args.n_hidden, 1, n_layer).to(DEVICE)
else:
    assert f"{net} is not supported!"

model.load_state_dict(torch.load(args.weights))
print(f"Loaded weights from {args.weights}")
model.eval()
with torch.no_grad():
    predicted = model(test_X)
rmse = calc_rmse(predicted, test_y)
mre  = calc_mre(predicted, test_y)
mad  = calc_mad(predicted, test_y)
pcc  = calc_pcc(predicted, test_y).item()

print('rmse:', calc_rmse(predicted, test_y))
print('mre:', calc_mre(predicted, test_y))
print('mad:', calc_mad(predicted, test_y))
print('pcc:', calc_pcc(predicted, test_y).item())

save_path = os.path.join(os.path.dirname(args.weights), 'eval_metrics.txt')
with open(save_path, 'w') as f:
    f.write(f'weights of model: {os.path.basename(args.weights)}\n')
    f.write(f'rmse: {rmse}\n')
    f.write(f'mre: {mre}\n')
    f.write(f'mad: {mad}\n')
    f.write(f'pcc: {pcc}\n')