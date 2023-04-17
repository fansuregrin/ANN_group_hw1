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

# norm data
if args.use_norm:
    training_X = F.normalize(training_X, dim=0)
    # training_y = F.normalize(training_y)
    test_X = F.normalize(test_X, dim=0)
    # test_y = F.normalize(test_y)

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