import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

kwargs = {
    "num_workers": 1,
    "pin_memory": True,
}

# Die Trainingsdaten aus dem NIST-Datensatz müssen runtergeladen werden
train_data = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1037,), (0.3081)),
            ],
        ),
    ),
    batch_size=64,
    shuffle=True,
    **kwargs,
)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081))],
        ),
    ),
    batch_size=64,
    shuffle=True,
    **kwargs,
)


# ----------------------- Training -----------------------
class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.convl = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conf_dropout = nn.Dropout2d()
        self.fcl1 = nn.Linear(320, 60)
        self.fcl2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.convl(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conf_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fcl1(x))
        x = self.fcl2(x)
        return x


model = Netz()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)


def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = data.cuda()
        target = target.cuda()
        # Variable() entfernt - nicht mehr nötig
        optimizer.zero_grad()
        out = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_id * len(data),
                len(train_data.dataset),
                100.0 * batch_id / len(train_data),
                loss.item(),
            )
        )


def test():
    model.eval()
    loss = 0
    correct = 0
    # torch.no_grad() statt volatile
    with torch.no_grad():
        for data, target in test_data:
            data = data.cuda()
            target = target.cuda()
            out = model(data)
            # reduction='sum' statt size_average=False, .item() statt .data[0]
            loss += F.cross_entropy(out, target, reduction='sum').item()
            prediction = out.data.max(1, keepdim=True)[1]
            correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()
    loss = loss / len(test_data.dataset)
    print("Durchschnittsloss: ", loss)
    print("Genauigkeit: ", 100.0 * correct / len(test_data.dataset))


for epoch in range(1, 10):
    train(epoch)
    test()