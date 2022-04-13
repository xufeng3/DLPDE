import torch
import torch.nn.functional as F

N = 40
d = 100

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(d, d+10)
        self.lin2 = torch.nn.Linear(d+10, d+10)
        self.lin3 = torch.nn.Linear(d+10, d)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x


class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.u0 = torch.nn.Parameter(torch.Tensor([50]))
        self.gu0 = torch.nn.Parameter(torch.rand(d, 1))
        self.u = None
        self.gu = None
        self.moduleList = torch.nn.ModuleList([Net() for _ in range(N)])

    def forward(self, x0, tlist):
        x = x0.view(-1)
        sigma = torch.diag(x) * 0.2
        u = self.u0
        gu = sigma @ self.gu0
        for i in range(N):
            deltat = tlist[i].view(1, 1)

            deltaW1 = torch.randn(size=(d, 1))
            deltaW = deltaW1 @ torch.sqrt(deltat)

            iu = torch.diag(torch.ones(d)*0.02)
            x = x.view(d, 1)
            miu = iu @ x
            miu = miu.view(d, 1)

            term2 = self.f(u) @ deltat
            term2 = term2.squeeze()

            term3 = gu.T @ deltaW
            term3 = term3.squeeze()

            u = u - term2 + term3

            sigma = torch.diag(x.squeeze()) * 0.2
            tmp = sigma @ deltaW
            tmp = tmp.view(d, 1)

            x = x + miu @ deltat + tmp

            self.gu = self.moduleList[i](x.T)
        return u

    def f(self, u):
        if u<50:
            # return torch.autograd.Variable(torch.Tensor([0.2]), requires_grad = True)
            return - 0.02 * u /3 - 0.02*u
        elif u>=70:
            # return torch.autograd.Variable(torch.Tensor([0.02]), requires_grad = True)
            return - (0.002)*u /3 - 0.02*u
        else:
            return - (70-50)/(0.02-0.2)*(u-50)/3  * u - 0.2/3  * u -0.02*u


pdeNet = Net2()
pdeOpt = torch.optim.Adam(pdeNet.parameters(), lr=1e-2)
epochNum = 300000

solutionList = []
for epoch in range(epochNum):
    # sample
    tList = torch.sort(torch.rand(size=(1, N - 1)))[0]
    tList1 = tList.numpy().tolist()[0]
    tList2 = [0] + tList1
    tList3 = tList1 + [1]
    deltaT_ = [tList3[i] - tList2[i] for i in range(N)]
    deltaT = torch.Tensor(deltaT_)
    x0 = torch.ones(size=(d, 1))*100
    gt = torch.Tensor([100])

    # begin training
    pdeNet.train()
    pdeOpt.zero_grad()
    output = pdeNet(x0, deltaT)
    loss = F.mse_loss(gt, output)
    loss.backward()
    pdeOpt.step()

    params = list(pdeNet.named_parameters())
    print("iteration = :", epoch, " ", params[0][1].item())
    solutionList.append(params[0][1].item())

print(solutionList)

# params = list(pdeNet.named_parameters())
# print(params.__len__())
# print(params[0][1])
# print(type(params[0][1].item()))

# # print the number of parameters (total and trainable parameters)
# total_params = sum(p.numel() for p in pdeNet.parameters())
# print(f'{total_params:,} total parameters.')
# total_trainable_params = sum(p.numel() for p in pdeNet.parameters() if p.requires_grad)
# print(f'{total_trainable_params:,} training parameters.')

import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.scatter(np.arange(len(solutionList)), solutionList)
plt.savefig('res.png')
plt.show()
