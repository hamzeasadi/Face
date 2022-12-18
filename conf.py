import os
import torch


root = os.path.join(os.getcwd(), 'data')
paths = dict(
    data=root, dataset=os.path.join(root, 'dataset'), haar=os.path.join(root, 'haar.xml'),
    train=os.path.join(root, 'dataset', 'train'), val=os.path.join(root, 'dataset', 'val'),
    pre_dataset=os.path.join(root, '105_classes_pins_dataset'), model=os.path.join(root, 'model')
)


def dist(a, p, n):
    ap = torch.dot(a, p)
    an = torch.dot(a, n)
    pn = torch.dot(p, n)
    print(ap+pn, an)


def main():
    a = torch.randn(size=(5, ))
    p = torch.randn(size=(5, ))
    n = torch.randn(size=(5, ))
    # dist(a, p, n)
    print(a)
    ap = torch.cross(a, p)
    print(ap)


if __name__ == '__main__':
    main()