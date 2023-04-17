import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import sys
from loguru import logger
from tensorboardX import SummaryWriter
from networks.models import Adaline, BP
from utils import seed_everything
from utils.ga import GA, GenomeReal, GenomeBinary, chrom_to_params, params_to_chrom


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='bp_question1', help='name for traing progress')
parser.add_argument('--n_hidden', type=int, help='number of neurons in hidden layer')
parser.add_argument('--n_layer', type=int, default=1, help='number of layers in network')
parser.add_argument('--net', type=str, default='bp', help='type of net: [bp, adaline]')
parser.add_argument('--activate', type=str, default='sigmoid', help='type of activation layer: [sigmoid, relu]')
parser.add_argument('--metric', type=str, default='mse', help='metric or loss function: [mse, mae]')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--start_iter', type=int, default=0, help='which iteration to start')
parser.add_argument('--n_iter', type=int, default=10000, help='number of iterations for training')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='dir to save model')
parser.add_argument('--n_iter_to_save', type=int, default=1000, help='number of iterations for saving model')
parser.add_argument('--use_norm', action='store_true', help='whether to norm data')
parser.add_argument('--pop_size', type=int, default=10, help='size of population')
parser.add_argument('--num_gen', type=int, default=100, help='number of evolutional generations')
args = parser.parse_args()

seed_everything(0)

with open('datasets/WindFarm_Data/Q1-wind farm data.txt', 'r') as f:
    lines = f.readlines()
training_X = [[float(line.split()[1]), float(line.split()[2])] for line in lines[4:124]]
training_y = [[float(line.split()[3])] for line in lines[4:124]]
test_X = [[float(line.split()[1]), float(line.split()[2])] for line in lines[124:155]]
test_y = [[float(line.split()[3])] for line in lines[124:155]]
training_X = torch.tensor(training_X)
training_y = torch.tensor(training_y)
test_X = torch.tensor(test_X)
test_y = torch.tensor(test_y)

name = args.name
net = args.net
activate = args.activate
metric = args.metric
lr = args.lr
start_iter  = args.start_iter
n_iteration = args.n_iter
n_layer = args.n_layer
checkpoint_dir = os.path.join(args.checkpoint_dir, name)
tensorboard_dir = os.path.join('question1_runs', name)

# initialize logger
log_dir = os.path.join(checkpoint_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
logger.add(sys.stdout, format="{time} {message}", filter='my_module', level='INFO')
logger.add(os.path.join(log_dir, "train_{time}.log"))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_X = training_X.to(DEVICE)
training_y = training_y.to(DEVICE)
test_X = test_X.to(DEVICE)
test_y = test_y.to(DEVICE)

tensorboard_writer = SummaryWriter(tensorboard_dir)

if net == 'bp':
    model = BP(2, args.n_hidden, 1, n_layer, activate).to(DEVICE)
elif net == 'adaline':
    model = Adaline(2, args.n_hidden, 1, n_layer).to(DEVICE)
else:
    assert f"{net} is not supported!"
# optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=lr)

if metric == 'mse':
    loss_fn = nn.MSELoss().to(DEVICE)
elif metric == 'mae':
    loss_fn = nn.L1Loss().to(DEVICE)
else:
    assert f"{metric} is not supported!"

logger.info('----------strat trainging--------')
for option, value in vars(args).items():
    logger.info(f'{option}: {value}')
logger.info(f'tensorboard_dir: {tensorboard_dir}')

if start_iter != 0:
    model_path = os.path.join(checkpoint_dir, f'{name}_{start_iter}.pth')
    model.load_state_dict(torch.load(model_path))
    logger.info(f"loaded model from {model_path}")

params_template = copy.deepcopy(model.state_dict())
chrom_len = 0
bound_l = np.empty(0)
bound_h = np.empty(0)
for key in params_template:
    param_length = np.prod(params_template[key].shape)
    if 'weight' in key:
        # kaiming uniform
        weight = params_template[key]
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        gain = torch.nn.init.calculate_gain('relu')
        _bound = gain * np.sqrt(3 / fan_in)
    elif 'bias' in key:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        _bound = 1 / np.sqrt(fan_in)
    else:
        raise Exception('Unknown parameter')
    bound_l = np.append(bound_l, -np.ones(param_length)*_bound)
    bound_h = np.append(bound_h, np.ones(param_length)*_bound)
    chrom_len += param_length
bound = np.array([bound_l, bound_h])


def train_model(model, loss_fn, optimizer, X, y, num_iter):
    loss_recorder = []
    for i in range(num_iter):
        model.train()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_recorder.append(loss.item())
    return np.mean(loss_recorder)

def calculate_fitness(chrom):
    params = chrom_to_params(chrom, params_template, DEVICE)
    model.load_state_dict(params)
    loss = train_model(model, loss_fn, optimizer, training_X, training_y, 1000)
    fitness = 1./loss
    return fitness

pop_size = args.pop_size
num_gen = args.num_gen
ga = GA(pop_size, chrom_len, bound, calculate_fitness, GenomeClass=GenomeBinary, cross_prob=0.5, mutate_prob=0.1)
ga.genetic(num_gen, logger=logger)

best_ga_params = chrom_to_params(ga.result(), params_template, DEVICE)
model.load_state_dict(best_ga_params)

model.train()
for i in range(start_iter, start_iter + n_iteration):
    training_y_hat = model(training_X)
    loss = loss_fn(training_y_hat, training_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1) % args.n_iter_to_save == 0:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{name}_latest.pth'))
    
    if (i+1) % 100 == 0 or i==0:
        # test
        model.eval()
        with torch.no_grad():
            test_y_hat = model(test_X)
        test_loss = loss_fn(test_y_hat, test_y)
        tensorboard_writer.add_scalars('loss', {'train_loss':loss, 'test_loss': test_loss}, i+1)
        logger.info(f"[iteration {i+1:06d}] train loss: {loss:.6f}, test loss: {test_loss:.6f}")

torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{name}_{start_iter + n_iteration}.pth'))