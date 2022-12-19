import os, sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import time
from torch import nn as nn
from torch import optim



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def euclidean_distance_matrix(x):
    """Efficient computation of Euclidean distance matrix
    Args:
    x: Input tensor of shape (batch_size, embedding_dim)

    Returns:
    Distance matrix of shape (batch_size, batch_size)
    """
    # step 1 - compute the dot product

    # shape: (batch_size, batch_size)
    dot_product = torch.mm(x, x.t())

    # step 2 - extract the squared Euclidean norm from the diagonal

    # shape: (batch_size,)
    squared_norm = torch.diag(dot_product)

    # step 3 - compute squared Euclidean distances

    # shape: (batch_size, batch_size)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)

    # get rid of negative distances due to numerical instabilities
    distance_matrix = F.relu(distance_matrix)

    # step 4 - compute the non-squared distances

    # handle numerical stability
    # derivative of the square root operation applied to 0 is infinite
    # we need to handle by setting any 0 to eps
    mask = (distance_matrix == 0.0).float()

    # use this mask to set indices with a value of 0 to eps
    eps = 1e-5
    distance_matrix += mask * eps

    # now it is safe to get the square root
    distance_matrix = torch.sqrt(distance_matrix)

    # undo the trick for numerical stability
    distance_matrix *= (1.0 - mask)

    return distance_matrix






def get_triplet_mask(labels):
    """compute a mask for valid triplets
    Args:
    labels: Batch of integer labels. shape: (batch_size,)
    Returns:
    Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j`, `k` are different.
    """
    # step 1 - get a mask for distinct indices

    # shape: (batch_size, batch_size)
    indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)
    # shape: (batch_size, batch_size, 1)
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    # shape: (1, batch_size, batch_size)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    # Shape: (batch_size, batch_size, batch_size)
    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # step 2 - get a mask for valid anchor-positive-negative triplets

    # shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    # shape: (batch_size, batch_size, 1)
    i_equal_j = labels_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_equal_k = labels_equal.unsqueeze(1)
    # shape: (batch_size, batch_size, batch_size)
    valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    # step 3 - combine two masks
    mask = torch.logical_and(distinct_indices, valid_indices)
    idxs = torch.where(mask==True)
    return idxs


class OrthoLoss(nn.Module):
    """
    doc
    """
    def __init__(self) -> None:
        super().__init__()
        self.l2 = nn.MSELoss()

    def triplet_loss(self, a, p, n):
        ap = torch.norm(a-p)
        an = torch.norm(a-n)
        costheta = torch.dot(a-p, a-n)/(an*ap)
        tantheta = torch.sqrt(1-torch.pow(costheta, 2))/costheta
        l21 = self.l2(ap/2, an*costheta) + self.l2(ap/tantheta, an*costheta/tantheta)
        # print(l21)
        return l21

    def trip2(self, a, p, n):
        ap = torch.norm(a-p)
        an = torch.norm(a-n)
        costheta = torch.dot(a-p, a-n)/(an*ap)
        tantheta = torch.sqrt(1-torch.pow(costheta, 2))/costheta
        return ap/tantheta

    def forward(self, embbedings, y):
        valid_triplet = get_triplet_mask(y)
        if valid_triplet[0].nelement() == 0:
            return torch.tensor([0.0], requires_grad=True)
        else:
            loss = self.triplet_loss(a=embbedings[valid_triplet[0][0]], p=embbedings[valid_triplet[1][0]], n=embbedings[valid_triplet[2][0]])
            # loss = self.trip2(a=embbedings[valid_triplet[0][0]], p=embbedings[valid_triplet[1][0]], n=embbedings[valid_triplet[2][0]])
            for i in range(valid_triplet[0].size()[0]):
                anchor_idx = valid_triplet[0][i]
                positive_idx = valid_triplet[1][i]
                negative_idx = valid_triplet[2][i]
                loss += self.triplet_loss(a=embbedings[anchor_idx], p=embbedings[positive_idx], n=embbedings[negative_idx])
                # loss = self.trip2(a=embbedings[valid_triplet[0][0]], p=embbedings[valid_triplet[1][0]], n=embbedings[valid_triplet[2][0]])
                # print(loss)
            return loss


class KeepTrack():
    
    def __init__(self, path) -> None:
        self.path = path
        self.state = dict(model="", opt="", epoch=1, minerror=0.1)

    def save_ckp(self, model: nn.Module, opt: optim.Optimizer, epoch, minerror, fname: str):
        self.state['model'] = model.state_dict()
        self.state['opt'] = opt.state_dict()
        self.state['epoch'] = epoch
        self.state['minerror'] = minerror
        save_path = os.path.join(self.path, fname)
        torch.save(obj=self.state, f=save_path)

    def load_ckp(self, fname):
        state = torch.load(os.path.join(self.path, fname), map_location=dev)
        return state



def main():
    batch_size = 5
    x = torch.randn(size=(batch_size, 5))
    labels = torch.randint(low=0, high=1, size=(batch_size, ))
    print(labels)

    valid_trip = get_triplet_mask(labels)
    if valid_trip[0].nelement() == 0:
        # print(len(valid_trip))
        print("empty")
        print(valid_trip)
    else:
        print("not empty")
        print(valid_trip)
    # dismtx = euclidean_distance_matrix(x)
    # print(dismtx)





if __name__ == '__main__':
    main()
