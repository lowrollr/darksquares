
import torch
import torch.nn as nn


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
    def __init__(self, input_layers, output_layers, residual_layers=20) -> None:
        super().__init__()
        self.conv1 = ConvolutionalLayer(input_layers, 128)

        self.residual_layers = []
        for _ in range(residual_layers):
            r = ResidualLayer(128, 128)
            self.residual_layers.append(r)
        self.conv2 = ConvolutionalLayer(128, output_layers, k_size=1, padding=0)
        self.passant_flatten = nn.Flatten()
        self.passant_layer = nn.Linear(64, 8)
        self.castle_flatten = nn.Flatten()
        self.castle_layer = nn.Linear(64, 2)

        self.loss_fn = nn.MSELoss()
        self.eight_bit_mask = 2 ** torch.arange(8 - 1, -1, -1).to(self.device, torch.float32)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def migrate_submodules(self):
        self.conv1 = self.conv1.to(self.device)
        self.conv2 = self.conv2.to(self.device)
        new_residual_layers = []
        for r in self.residual_layers:
            new_residual_layers.append(r.to(self.device))
        self.residual_layers = new_residual_layers
        

    def forward(self, x):
        absences = torch.select_copy(x, dim=1, index=0).unsqueeze(1).expand((-1,6,-1,-1))
        piece_invariant = torch.sum(self.eight_bit_mask * x[:,4,0,:], -1)
        x = self.conv1(x)
        for r in self.residual_layers:
            x = r(x)
        x = self.conv2(x) 
        x = torch.sigmoid(x)
        
        # normalize first 6 layers to sum to PI 
        probs, passant, castle = torch.split(x, (6, 1, 1), dim=1)
        # zero out any squares in probs where we know there isn't an opponet's piece
        probs = probs * (-absences + 1)
        castle = self.castle_flatten(castle)
        passant = self.passant_flatten(passant)

        castle = torch.sigmoid(self.castle_layer(castle))
        passant = torch.sigmoid(self.passant_layer(passant))
        probs = probs * (piece_invariant / torch.sum(probs, (3,2,1))).view(-1,1,1,1)
        return probs, passant, castle

    def loss_fn(self, input, output, actual) -> torch.TensorType:
        print('called loss fn!')
        # slice input to yield the same as expected output
        input = input[:,14:20,:,:], input[:,20,:,0], input[:,21,0,3:5]
        print([a.shape for a in actual])
        return sum([self.loss_fn(a, b) for a,b in zip(output, actual)]) - sum([self.loss_fn(a, b) for a,b in zip(input, actual)])
