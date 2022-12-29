
import torch
import torch.nn as nn
import numpy as np

class ResidualLayer(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        
    def forward(self, x):
        x = nn.functional.relu(self.block(x) + x)
        return x

class ConvolutionalLayer(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, padding=1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

WEIGHTS = torch.from_numpy(np.stack([np.full((8,8), 18), np.full((8,8),9), np.full((8,8),5), np.full((8,8),3), np.full((8,8),3), np.full((8,8),1)]))    

def weighted_mse_loss(input, target):
    return torch.mean(WEIGHTS * ((input - target) ** 2))

class BeliefNet(nn.Module):
    def __init__(self, input_layers, output_layers, residual_layers=20) -> None:
        super().__init__()
        self.conv1 = ConvolutionalLayer(input_layers, 128)
        layers = []
        for _ in range(residual_layers):
            r = ResidualLayer(128, 128)
            layers.append(r)
        self.residual_layers = nn.Sequential(*layers)
        self.conv2 = ConvolutionalLayer(128, output_layers, k_size=1, padding=0)
        self.passant_flatten = nn.Flatten()
        self.passant_layer = nn.Linear(64, 8)
        self.castle_flatten = nn.Flatten()
        self.castle_layer = nn.Linear(64, 2)
        # self.mse_loss = nn.MSELoss()
        self.mse_loss = nn.MSELoss()
        
    
    @property
    def device(self):
        return next(self.parameters()).device

    def migrate_submodules(self):
        self.conv1 = self.conv1.to(self.device)
        self.conv2 = self.conv2.to(self.device)
        new_residual_layers = []
        for r in self.residual_layers:
            new_residual_layers.append(r.to(self.device))
        self.residual_layers = nn.Sequential(*new_residual_layers)
        

    def forward(self, x):
        x = self.conv1(x)
        for r in self.residual_layers:
            x = r(x)
        x = self.conv2(x) 
        x = torch.sigmoid(x)

        probs, passant, castle = torch.split(x, (6, 1, 1), dim=1)
        castle = torch.sigmoid(self.castle_layer(self.castle_flatten(castle)))
        passant = torch.sigmoid(self.passant_layer(self.passant_flatten(passant)))
        return probs, passant, castle

    def loss_fn(self, inp, output, actual) -> torch.TensorType:
        # slice input to yield the same as expected output
        inp_pieces, inp_passant, inp_castle = inp[:,14:20,:,:], inp[:,20,:,0], inp[:,21,0,3:5]
        act_pieces, act_passant, act_castle = actual
        out_pieces, out_passant, out_castle = output
        out_loss = weighted_mse_loss(out_pieces, act_pieces) + self.mse_loss(out_passant, act_passant) + self.mse_loss(out_castle, act_castle)
        inp_loss = weighted_mse_loss(inp_pieces, act_pieces) + self.mse_loss(inp_passant, act_passant) + self.mse_loss(inp_castle, act_castle)
        return out_loss - inp_loss


        # return sum([self.mse_loss(a, b) for a,b in zip(output, actual)]) \
          #      - sum([self.mse_loss(a, b) for a,b in zip(inp, actual)])
