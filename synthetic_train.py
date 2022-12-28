


import numpy as np
import torch
import os
from net import BeliefNet


EPOCHS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BeliefNet(22,8).to(device)
model.migrate_submodules()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch = 0
loss = None



# load model
if os.path.exists('model.pt'):
    checkpoint = torch.load('model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']





# load inputs
inputs = np.load('inputs.npy', allow_pickle=True)
# load actuals
actual_boards = np.load('actual_boards.npy', allow_pickle=True)
actual_en_passants = np.load('actual_en_passants.npy', allow_pickle=True)
actual_castles = np.load('actual_castles.npy', allow_pickle=True)

input_tensor = torch.from_numpy(inputs)
board_tensor = torch.from_numpy(actual_boards)
passant_tensor = torch.from_numpy(actual_en_passants)
castling_tensor = torch.from_numpy(actual_castles)

dataset = torch.utils.data.TensorDataset(input_tensor, board_tensor, passant_tensor, castling_tensor)

seed = torch.Generator().manual_seed(42)

proportions = [.9, .10]
lengths = [int(p * len(dataset)) for p in proportions]
lengths[-1] = len(dataset) - sum(lengths[:-1])

train, test = torch.utils.data.random_split(dataset, lengths, generator=seed)

train_loader = torch.utils.data.DataLoader(train, batch_size=1024, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=1024, shuffle=True)

while epoch < EPOCHS:
    for input_batch, board_batch, passant_batch, castling_batch in train_loader:
        input_batch = input_batch.to(device)
        board_batch = board_batch.to(device)
        passant_batch = passant_batch.to(device)
        castling_batch = castling_batch.to(device)

        optimizer.zero_grad()

        out = model(input_batch)
        loss = model.loss_fn(input_batch, out, (board_batch, passant_batch, castling_batch))
        loss.backward()
        print('loss:', loss.item())
        optimizer.step()

    with torch.no_grad():
        for val_input, val_board, val_passant, val_castling in test_loader:
            val_input = val_input.to(device)
            val_board = val_board.to(device)
            val_passant = val_passant.to(device)
            val_castling = val_castling.to(device)

            val_out = model(val_input)
            val_loss = model.loss_fn(val_input, val_out, (val_board, val_passant, val_castling))
            print('val_loss:', val_loss.item())
    epoch += 1

            
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'model.pt')











