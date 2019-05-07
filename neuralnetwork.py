import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(9,10) # 9 inputs - 18 outpus for hidden layer
        self.fc2 = nn.Linear(10,20) # 18 inputs hidden
        self.fc3 = nn.Linear(20,10)
        self.fc4 = nn.Linear(10,9) # 18 inputs hidden layer 9 output

    def forward(self, x):
        x = Variable(torch.FloatTensor(x), requires_grad = True)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = F.relu(self.fc4(x))
        return y

def returnNet():
    net = Net()
    try:
        net.load_state_dict(torch.load("rednn"))
    except:
        pass
    return net

def transformData(a):
    b=[]
    b.append([a[0],a[1]])
    b.append([[a[0][6],a[0][3],a[0][0],a[0][7],a[0][4],a[0][1],a[0][8],a[0][5],a[0][2]],\
             [a[1][6],a[1][3],a[1][0],a[1][7],a[1][4],a[1][1],a[1][8],a[1][5],a[1][2]]])
    b.append([[a[0][8],a[0][7],a[0][6],a[0][5],a[0][4],a[0][3],a[0][2],a[0][1],a[0][0]],\
             [a[1][8],a[1][7],a[1][6],a[1][5],a[1][4],a[1][3],a[1][2],a[1][1],a[1][0]]])
    b.append([[a[0][2],a[0][5],a[0][8],a[0][1],a[0][4],a[0][7],a[0][0],a[0][3],a[0][6]],\
             [a[1][2],a[1][5],a[1][8],a[1][1],a[1][4],a[1][7],a[1][0],a[1][3],a[1][6]]])
    b.append([[a[0][2],a[0][1],a[0][0],a[0][5],a[0][4],a[0][3],a[0][8],a[0][7],a[0][6]],\
             [a[1][2],a[1][1],a[1][0],a[1][5],a[1][4],a[1][3],a[1][8],a[1][7],a[1][6]]])
    b.append([[a[0][0],a[0][3],a[0][6],a[0][1],a[0][4],a[0][7],a[0][2],a[0][5],a[0][8]],\
             [a[1][0],a[1][3],a[1][6],a[1][1],a[1][4],a[1][7],a[1][2],a[1][5],a[1][8]]])
    b.append([[a[0][6],a[0][7],a[0][8],a[0][3],a[0][4],a[0][5],a[0][0],a[0][1],a[0][2]],\
             [a[1][6],a[1][7],a[1][8],a[1][3],a[1][4],a[1][5],a[1][0],a[1][1],a[1][2]]])
    b.append([[a[0][8],a[0][5],a[0][2],a[0][7],a[0][4],a[0][1],a[0][6],a[0][3],a[0][0]],\
             [a[1][8],a[1][5],a[1][2],a[1][7],a[1][4],a[1][1],a[1][6],a[1][3],a[1][0]]])
    return b

def trainingNet(data, net):
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),  lr = 0.01, momentum = 0.5)
    for epoch in range(200):
        for i, data_e in enumerate(data):
            data_e = transformData(data_e)
            for l, data2 in enumerate(data_e):
                X, Y = iter(data2)
                X, Y = [X], Variable(torch.FloatTensor([Y]), requires_grad = False)
                optimizer.zero_grad()
                outputs = net(X)
                loss = criterion(outputs, Y)
                loss.backward()
                optimizer.step()

def firstTraining(net):
    data = [((1,1,0,-1,-1,0,0,0,0), (1,0,0,0,0,0,0,0,0)),
            ((1,0,1,0,0,0,0,-1,-1), (0,1,0,0,0,0,0,0,0)),
            ((-1,-1,0,0,0,0,1,0,0), (0,0,1,0,0,0,0,0,0)),
            ((0,0,1,0,0,0,0,-1,-1), (0,0,0,0,0,0,1,0,0)),
            ((-1,0,1,0,0,0,0,0,-1), (0,0,0,0,1,0,0,0,0)),
            ((0,0,0,0,1,0,-1,1,-1), (0,1,0,0,0,0,0,0,0)),
            ((-1,0,0,0,-1,0,1,0,0), (0,0,1,0,0,0,0,0,1))]
    trainingNet(data,net)

def saveNet(net):
    torch.save(net.state_dict(), "rednn")
