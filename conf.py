import os


root = os.path.join(os.getcwd(), 'data')
paths = dict(
    data=root, dataset=os.path.join(root, 'dataset'), haar=os.path.join(root, 'haar.xml'),
    train=os.path.join(root, 'dataset', 'train'), val=os.path.join(root, 'dataset', 'val'),
    pre_dataset=os.path.join(root, '105_classes_pins_dataset'), model=os.path.join(root, 'model')
)

def main():
    pass



if __name__ == '__main__':
    main()