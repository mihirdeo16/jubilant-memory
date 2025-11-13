import torch

class LoraLayer(torch.nn.Module):
    def __init__(self,in_dim,out_dim,alpha,rank):
        super().__init__()

        self.alpha = alpha

        self.A = torch.nn.Parameter(torch.randint(in_dim,rank))
        self.B = torch.nn.Parameter(torch.zeros(rank,out_dim))

    def forward(self):

        # Operation as: self.alpha * (self.A @ self.B)

        return torch.multiply(self.alpha,torch.matmul(self.A,self.B))