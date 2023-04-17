import torch.nn as nn


class Adaline(nn.Module):
    def __init__(self, in_f, n_hidden, out_f, n_layer=1):
        super().__init__()
        layers = [
            nn.Linear(in_f, n_hidden)
        ]
        if n_layer > 1:
            layers += ([nn.Linear(n_hidden, n_hidden),] * (n_layer-1))
        layers.append(nn.Linear(n_hidden, out_f))
        self.net = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.net(X)
    

class BP(nn.Module):
    def __init__(self, in_f, n_hidden, out_f, n_layer=1, activate='sigmoid'):
        super().__init__()
        if activate == 'sigmoid':
            activation = nn.Sigmoid
        elif activate == 'relu':
            activation = nn.ReLU
        else:
            assert f'{activate} is not supported!'
        layers = [
            nn.Linear(in_f, n_hidden),
            activation(),
        ]
        if n_layer > 1:
            layers += ([nn.Linear(n_hidden, n_hidden), activation()] * (n_layer -1))
        layers.append(nn.Linear(n_hidden, out_f))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)
    

if __name__ == '__main__':
    m = BP(2, 10, 1, 3)
    print(m)