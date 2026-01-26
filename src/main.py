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
        criterion = nn.CrossEntropyLoss # hier geht auch SoftMax
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

for epoch in range(1, 30):
    train(epoch)