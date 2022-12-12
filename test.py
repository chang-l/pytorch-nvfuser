import torch
import torch.nn as nn

torch._C._debug_set_autodiff_subgraph_inlining(False)
# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.indx = torch.randint(10, (10,)).to('cuda')

    def forward(self, x):
        x = self.fc1(x)
        x = torch.index_select(x, 0, self.indx)
        x = torch.permute(x, [1,0])
        x = x * 4
        x = x + 20
        x = torch.relu(x)
        return x

# Instantiate the neural network
net = SimpleNet().to('cuda')
net = torch.jit.script(net)

# Define an input tensor
x = torch.randn(10, 10).to('cuda')

# Forward pass
output = net(x)

for i in range(4):
    output = net(x)

print(net.graph_for())
