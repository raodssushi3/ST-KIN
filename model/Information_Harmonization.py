import torch
import torch.nn as nn


class fusion(nn.Module):
    def __init__(self, dim):
        super(fusion, self).__init__()
        self.fc_zonghe = nn.Linear(dim, dim, bias=True)
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)


    def forward(self, x1, x2):
        x1 = torch.transpose(x1,1,2)
        x2 = torch.transpose(x2,1,2)
        x11 = x1.clone()
        x21 = x2.clone()

        x = x1 + x2
        x = torch.sigmoid(self.fc_zonghe(x))
        x11 = torch.tanh(self.fc1(x11))
        x21 = torch.tanh(self.fc1(x21))
        x1_out = x*x11
        x2_out = (1-x)*x21
        out1 = torch.transpose(x1_out, 1, 2)
        out2 = torch.transpose(x2_out, 1, 2)

        out = out1 + out2
        return out

