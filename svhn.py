from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data
import scipy.io as sio
import pdb
import torch.nn.functional as F
import torch

class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """


    def __init__(self, root, split='train', transform=None, target_transform=None):
        # super(SVHN, self).__init__(root, transform=transform,
        #                            target_transform=target_transform)
        split_list = {"train":"train" , "test":"test" , "extra_svhn":"extra", "extra_synth":"synth_train" , "svhn_aug":"svhn_augmented" , "syn_aug":"syndigit_augmented",
                            "noise_1p":"svhn_noiseaug_1p" , "noise_5p":"svhn_noiseaug_5p" , 
                                "noise_10p":"svhn_noiseaug_10p" , "noise_20p":"svhn_noiseaug_20p"}

        assert split in split_list

        self.root = root

        self.transform = transform
        self.target_transform = target_transform

        self.split = split# verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.filename = f"{split_list[self.split]}_32x32.mat" # Split should be one of "train", "test" , "extra_svhn" , "extra_emnist"

        
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # self.pseudo_labels = loaded_mat["pseudo_y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if img.shape[0] == 1:
            img = Image.fromarray(np.squeeze(img) , mode='L')
        else:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if img.shape[0] == 1: ## Different treatment for EMNIST
            img = F.pad(img , (2,2,2,2))
            img = torch.cat(3*[img])

        return img, target

    def __len__(self):
        return len(self.data)


    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)
