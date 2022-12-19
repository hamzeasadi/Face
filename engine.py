import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from utils import euclidean_distance_matrix
from sklearn.cluster import KMeans


# dev
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(model: nn.Module, data: DataLoader, criterion: nn.Module, optimizer: optim):
    epoch_error = 0
    l = len(data)
    model.train()
    for i, (X, Y) in enumerate(data):
        out = model(X.to(dev))
        loss = criterion(out, Y.to(dev))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_error += loss.item()
        # break
    return epoch_error/l


def val_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            out = model(X.to(dev))
            loss = criterion(out, Y.to(dev))
            epoch_error += loss.item()

    return epoch_error/l


def test_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    out = 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            out = model(X)
            # loss = criterion(out, Y)
            # epoch_error += loss.item()
            break

    X = out.cpu().detach().numpy()
    print(X.shape)
    Y = Y.cpu().detach().numpy()
    # print(f"test-loss={epoch_error}")
    

    # kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    # y = Y.numpy()
    # colors = ['green', 'blue', 'red', 'yellow', 'cyan', 'orange', 'black', 'magenta']
    # color_code = []
    # # for i in range(len(y)):
    # #     color_code.append(colors[y[i]])
    # # print(color_code)
    # # x = np.random.randint(low=1, high=10, size=(len(y), 2))
    # X = out.numpy()
    # X_embedded = TSNE(n_components=2, learning_rate='auto',
    #                 init='random', perplexity=3).fit_transform(X)
    # print(X_embedded.shape)

    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    # plt.savefig('output.png')
    # plt.show()



def main():
    pass



if __name__ == '__main__':
    main()