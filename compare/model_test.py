import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from compare import get_plot_data, read_csv, get_image_path


class net(nn.Module):
    def __init__(self, n):
        super().__init__()
        h = n * 2
        self.net = nn.Sequential(
            nn.Linear(n, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        Y_pred = self.forward(x)
        return Y_pred


class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


def fit_v2(x, y, model, opt, loss_fn, epochs=1000):
    for epoch in range(epochs):
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(loss.item())

    return loss.item()


def main():
    import algorithms

    csv_path = "../exp14/white.csv"
    images_path = "../exp14/images/White"
    data = read_csv(csv_path)
    N = 10

    image_names = list(data)
    x = [
        algorithms.nheights.image_to_independent(
            get_image_path(n, dir=images_path), threshold=50, nheights=N
        )
        for n in image_names
    ]
    # x = [(algorithms.maxheight.image_to_independent(get_image_path(n, dir=images_path), threshold=50), algorithms.nheights_area.image_to_independent(get_image_path(n, dir=images_path), threshold=50, nheights=10)) for n in image_names]
    y = [data[n] for n in image_names]

    d = dataset(x, y)

    model = net(N)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    # optimizer = torch.optim.Adam(model.parameters(),lr=0.5)
    epochs = 1000

    from functools import partial

    X_train, Y_train = map(partial(torch.tensor, dtype=torch.float32), (x, y))
    print(X_train.dtype, Y_train.dtype)

    print(
        "Final loss",
        fit_v2(X_train, Y_train, model, optimizer, F.mse_loss, epochs=epochs),
    )
    n = "250uL_1.jpg"
    to_predict = algorithms.nheights.image_to_independent(
        get_image_path(n, dir=images_path), threshold=50, nheights=N
    )
    p = model.predict(torch.tensor(to_predict, dtype=torch.float32)).item()
    a = data[n]
    print(p, a)


if __name__ == "__main__":
    main()
