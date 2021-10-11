from torch import nn
import torch


class sc_up_fc_nn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_array=[], functions_array=[]):
        super().__init__()
        self.linear_functions = []
        self.functions_array = [x() for x in functions_array]
        self.hidden_layers = len(hidden_dim_array)
        for index in range(self.hidden_layers):
            self.linear_functions.append(nn.Linear(input_dim, hidden_dim_array[index]))
            input_dim = hidden_dim_array[index]
        self.linear_functions = nn.ModuleList(self.linear_functions)
        self.final_linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = x
        for i in range(self.hidden_layers):
            out = self.linear_functions[i](out)
            out = self.functions_array[i](out)
            out = self.dropout(out)
        out = self.final_linear(out)
        return torch.sigmoid(out)


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.shape[0], -1)
        target = target.view(target.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100 * batch_idx / len(train_loader), loss.item()))
    return
