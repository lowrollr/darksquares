
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




class BeliefNet(nn.Module):
    def __init__(self, 
                input_layers, main_layers=15, head_layers=5) -> None:
        super().__init__()
        
        layers = [ConvolutionalLayer(input_layers, 128)]
        for _ in range(main_layers):
            r = ResidualLayer(128, 128)
            layers.append(r)
        layers.append(ConvolutionalLayer(128, 128))
        self.main_vein = nn.Sequential(*layers)
        
        board_layers = []
        pieces_layers = []
        ep_layers = []
        castle_layers = []
        for _ in range(head_layers):
            board_layers.append(ResidualLayer(128,128))
            pieces_layers.append(ResidualLayer(128,128))
            ep_layers.append(ResidualLayer(128,128))
            castle_layers.append(ResidualLayer(128,128))

        board_layers.append(ConvolutionalLayer(128,7, k_size=1, padding=0))
        board_layers.append(nn.Sigmoid())
        board_layers.append(nn.Softmax(dim=1))

        pieces_layers.append(ConvolutionalLayer(128,5, k_size=1, padding=0))
        pieces_layers.append(nn.Flatten(start_dim=2))
        pieces_layers.append(nn.Linear(64,8))
        pieces_layers.append(nn.Sigmoid())
        pieces_layers.append(nn.Softmax(dim=1))
        
        ep_layers.append(ConvolutionalLayer(128,1, k_size=1, padding=0))
        ep_layers.append(nn.Flatten(start_dim=1))
        ep_layers.append(nn.Linear(64,9))
        ep_layers.append(nn.Sigmoid())
        ep_layers.append(nn.Softmax(dim=1))

        castle_layers.append(ConvolutionalLayer(128,1, k_size=1, padding=0))
        castle_layers.append(nn.Flatten(start_dim=1))
        castle_layers.append(nn.Linear(64,2))
        castle_layers.append(nn.Sigmoid())

        self.board_layers = nn.Sequential(*board_layers)
        self.pieces_layers = nn.Sequential(*pieces_layers)
        self.ep_layers = nn.Sequential(*ep_layers)
        self.castle_layers = nn.Sequential(*castle_layers)

        # self.mse_loss = nn.MSELoss()
        self.mse_loss = nn.MSELoss()


    def weighted_mse_loss(self, input, target):
        return torch.mean(self.WEIGHTS * ((input - target) ** 2))

    @property
    def device(self):
        return next(self.parameters()).device

    def migrate_submodules(self):
        self.main_vein = self.main_vein.to(self.device)
        self.board_layers = self.board_layers.to(self.device)
        self.pieces_layers = self.pieces_layers.to(self.device)
        self.ep_layers = self.ep_layers.to(self.device)
        self.castle_layers = self.castle_layers.to(self.device)


    def forward(self, x):
        x = self.main_vein(x)

        board = self.board_layers(x)
        pieces = self.pieces_layers(x)
        ep = self.ep_layers(x)
        castle = self.castle_layers(x)

        return board, pieces, ep, castle


    def loss_fn(self, inp, output, actual) -> torch.TensorType:
        # slice input to yield the same as expected output
        inp = inp[:,14:20,:,:], inp[:,20,:,0], inp[:,21,0,3:5], inp[:,4,0:5,:]
        
        return sum([self.mse_loss(a, b) for a,b in zip(output, actual)]) \
               - sum([self.mse_loss(a, b) for a,b in zip(inp, actual)])
