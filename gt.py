import torch
import torch.nn as nn


d = torch.device('cpu')
g = torch.ones([10,10],requires_grad=True)
k = torch.ones([10,1])
ll = nn.Linear(10,1).to(d)
l = g.to(d)
k = k.to(d)

ppn = [p for p in ll.parameters()] + [l]
optimizer = torch.optim.SGD(ppn, lr=0.01)
for i in range(20):
    f = ll(l).mean()
    optimizer.zero_grad()
    f.backward()
    optimizer.step()
    print(l)