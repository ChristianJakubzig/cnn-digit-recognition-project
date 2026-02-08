import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

kwargs = {'num_workers' : 1, 'pin_memory' : True} # Wird benötigt wenn man die Rechenprozesse mit der GPU Beschleunigen möchte

# Die Trainingsdaten aus dem NIST-Datensatz müssen runtergeladen werden
train_data = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=True, # Es handelt sich um die Trainingsdaten
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081))], # Die Daten werden als Tensoren gespeichert da unser Netz damit trainiert wird. Die Tensoren müssen noch normalisiert werden.
        ),
    ), batch_size=64, shuffle=True, **kwargs # Die Batch_size gibt an wieviel auf einmal verarbeitet werden kann. Leistungs stärkerer Rechner = höhere batsh_size. Shuffel -> Damit die Daten dem Netz nicht immer in der selben Reihenfolge präsentiert werden
)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False, # Es handelt sich um die Testdaten
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081))],
        ),
    ), batch_size=64, shuffle=True, **kwargs
)

# ----------------------- Training -----------------------
class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.convl = nn.Conv2d(1, 10, kernel_size=5) # Erste Schicht convolutional Layer, es werden immer 25 Pixel zusammengefasst auf einen 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # Zweite schicht -> Nochmal mit dem Output aus 1
        self.conf_dropout = nn.Dropout2d() # Hiermit können Pixel vergessen werden. Sonst lernt das Netz nur auswendig
        self.fcl1 = nn.Linear(320, 60)
        self.fcl2 = nn.Linear(60, 10) # zehn Outputs da wir 10 Möglichkeiten haben was erkannt werden kann

    def forward(self, x):
        x = self.convl(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x) # aktivierungsfunktion die besonders für convolutional Layers gut ist
        x = self.conv2(x)
        x = self.conf_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # -------Schritt zum testen ----------
        # print(x.size()) # Ergebnis = torch.Size([64, 20, 4, 4]) 64 = Bilder da Batshsize = 64, 20 Haben wir als Output festgelegt in conv2, 4X4 ist die Bildgröße in die wir aufgeteilt haben -> 20 x 4 x 4 = 320 deswegen 320 in fcl1
        # exit()
        x = x.view(-1, 320)
        x = F.relu(self.fcl1(x))
        x = self.fcl2(x)
        # return F.log_softmax(x, dim=1) # es gibt nur einen Gewinner alle anderen werden auf 0 gesetzt (geht irgendwie nicht)
        return x

model = Netz()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)

def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = data.cuda()
        target = target.cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad() # setzt alle Gradienten auf Null -> Muss jedes mal passieren / Aktivierungsfunktion
        out = model(data)
        criterion = nn.CrossEntropyLoss() # hier geht auch SoftMax
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_data.dataset),
            100. * batch_id / len(train_data), loss.item()
        ))

for epoch in range(1, 30):
    train(epoch)