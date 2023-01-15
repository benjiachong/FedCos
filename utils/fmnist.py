import os
import gzip
import numpy as np
import torch.utils.data as data
from torchfusion.datasets import datasets

'''
class FMNIST(data.Dataset):
    def __init__(self):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
'''
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)


    labels = torch.from_numpy(labels).type(torch.LongTensor)
    images = torch.from_numpy(images).type(torch.float32)


    dataset = torch.utils.data.TensorDataset(images, labels)

    return dataset


if __name__ == '__main__':
    dloader = datasets.fashionmnist_loader(size=28,batch_size=32)
    dataset = FashionMNIST(root,train=train,transform=trans,download=download,target_transform=target_transform)
    print(1)