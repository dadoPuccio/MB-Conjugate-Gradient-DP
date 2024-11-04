import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from sklearn import metrics

def get_dataset(dataset_name, train_flag, datadir, exp_dict, device):
    if dataset_name == "mnist":
        class MNISTindex(Dataset):
            def __init__(self):
                self.dataset = torchvision.datasets.MNIST(datadir, train=train_flag,
                                       download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.5,), (0.5,))
                                       ]))

            def __getitem__(self, index):
                data, target = self.dataset.__getitem__(index)
                return data, target, index

            def __len__(self):
                return len(self.dataset)

        dataset = MNISTindex()


    if dataset_name == "cifar10":
        class CIFAR10index(Dataset):
            def __init__(self):
                
                if train_flag:
                    transform_function = transforms.Compose([
                        transforms.RandomAffine(0.0, translate=(0.1, 0.1), shear=0, fill=0), # new # A
                        transforms.RandomCrop(32, padding=4), # C
                        transforms.RandomHorizontalFlip(), # H
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010)),
                    ])
                    transform_function.transforms.insert(0, transforms.RandAugment(1,1)) # new # T
                else:
                    transform_function = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010)),
                    ])

                self.dataset = torchvision.datasets.CIFAR10(
                    root=datadir,
                    train=train_flag,
                    download=True,
                    transform=transform_function)

            def __getitem__(self, index):
                data, target = self.dataset.__getitem__(index)
                return data, target, index

            def __len__(self):
                return len(self.dataset)

        dataset = CIFAR10index()


    if dataset_name == "cifar100":
        class CIFAR100index(Dataset):
            def __init__(self):

                if train_flag:
                    transform_function = transforms.Compose([
                        transforms.RandomAffine(0.0, translate=(0.1, 0.1), shear=0, fill=0), # new # A
                        transforms.RandomCrop(32, padding=4), # C
                        transforms.RandomHorizontalFlip(), # H
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010)),
                    ])
                    transform_function.transforms.insert(0, transforms.RandAugment(1,1)) # new # T
                else:
                    transform_function = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010)),
                    ])

                self.dataset = torchvision.datasets.CIFAR100(
                    root=datadir,
                    train=train_flag,
                    download=True,
                    transform=transform_function)

            def __getitem__(self, index):
                data, target = self.dataset.__getitem__(index)
                return data, target, index

            def __len__(self):
                return len(self.dataset)

        dataset = CIFAR100index()


    if dataset_name == "fashion":
        class FashionMNISTindex(Dataset):
            def __init__(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5), (0.5))
                                                ])

                # Download and load the training data
                self.dataset = torchvision.datasets.FashionMNIST(datadir, download=True, train=train_flag, transform=transform)

            def __getitem__(self, index):
                data, target = self.dataset.__getitem__(index)
                return data, target, index

            def __len__(self):
                return len(self.dataset)

        dataset = FashionMNISTindex()


    if dataset_name in ["mushrooms", "rcv1", "ijcnn"]:
        sigma_dict = {"mushrooms": 0.5,
                      "rcv1":0.5,
                      "ijcnn":0.05}

        X, y = load_libsvm(dataset_name, data_dir=datadir)

        labels = np.unique(y)
        y[y==labels[0]] = 0
        y[y==labels[1]] = 1
        # splits used in experiments
        splits = train_test_split(X, y, test_size=0.2, shuffle=True,
                    random_state=9513451)
        X_train, X_test, Y_train, Y_test = splits

        if train_flag:
            fname_rbf = "%s/rbf_%s_%s_train.npy" % (datadir, dataset_name, sigma_dict[dataset_name])
            if os.path.exists(fname_rbf):
                k_train_X = np.load(fname_rbf)
            else:
                k_train_X = rbf_kernel(X_train, X_train, sigma_dict[dataset_name])
                np.save(fname_rbf, k_train_X)
                print('%s saved' % fname_rbf)

            X_train = k_train_X
            X_train = torch.FloatTensor(X_train)
            Y_train = torch.LongTensor(Y_train)
            indexes = torch.FloatTensor(np.arange(len(Y_train)))

            dataset = torch.utils.data.TensorDataset(X_train, Y_train, indexes)

        else:
            fname_rbf = "%s/rbf_%s_%s_test.npy" % (datadir, dataset_name, sigma_dict[dataset_name])
            if os.path.exists(fname_rbf):
                k_test_X = np.load(fname_rbf)
            else:
                k_test_X = rbf_kernel(X_test, X_train, sigma_dict[dataset_name])
                np.save(fname_rbf, k_test_X)
                print('%s saved' % fname_rbf)

            X_test = k_test_X
            X_test = torch.FloatTensor(X_test)
            Y_test = torch.LongTensor(Y_test)
            indexes = torch.FloatTensor(np.arange(len(Y_test)))

            dataset = torch.utils.data.TensorDataset(X_test, Y_test, indexes)

    return dataset


# ===========================================================
# Helpers
import os
import urllib

import numpy as np
from sklearn.datasets import load_svmlight_file


LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1"       : "rcv1_train.binary.bz2",
                      "mushrooms"  : "mushrooms",
                      "ijcnn"      : "ijcnn1.tr.bz2"}


def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return X, y


def rbf_kernel(A, B, sigma):
    distsq = np.square(metrics.pairwise.pairwise_distances(A, B, metric="euclidean"))
    K = np.exp(-1 * distsq/(2*sigma**2))
    return K

from sklearn.svm import SVC

def make_binary_linear(n, d, margin, y01=False, bias=False, separable=True, scale=1, shuffle=True, seed=None):
    # Copied from https://github.com/IssamLaradji/sps
    assert margin >= 0.

    if seed:
        np.random.seed(seed)

    labels = [-1, 1]

    # Generate support vectors that are 2 margins away from each other
    # that is also linearly separable by a homogeneous separator
    w = np.random.randn(d); w /= np.linalg.norm(w)
    # Now we have the normal vector of the separating hyperplane, generate
    # a random point on this plane, which should be orthogonal to w
    p = np.random.randn(d-1); l = (-p@w[:d-1])/w[-1]
    p = np.append(p, [l])

    # Now we take p as the starting point and move along the direction of w
    # by m and -m to obtain our support vectors
    v0 = p - margin*w
    v1 = p + margin*w
    yv = np.copy(labels)

    # Start generating points with rejection sampling
    X = []; y = []
    for i in range(n-2):
        s = scale if np.random.random() < 0.05 else 1

        label = np.random.choice(labels)
        # Generate a random point with mean at the center 
        xi = np.random.randn(d)
        xi = (xi / np.linalg.norm(xi))*s

        dist = xi@w
        while dist*label <= margin:
            u = v0-v1 if label == -1 else v1-v0
            u /= np.linalg.norm(u)
            xi = xi + u
            xi = (xi / np.linalg.norm(xi))*s
            dist = xi@w

        X.append(xi)
        y.append(label)

    X = np.array(X).astype(float); y = np.array(y)#.astype(float)

    if shuffle:
        ind = np.random.permutation(n-2)
        X = X[ind]; y = y[ind]

    # Put the support vectors at the beginning
    X = np.r_[np.array([v0, v1]), X]
    y = np.r_[np.array(yv), y]

    if separable:
        # Assert linear separability
        # Since we're supposed to interpolate, we should not regularize.
        clff = SVC(kernel="linear", gamma="auto", tol=1e-10, C=1e10)
        clff.fit(X, y)
        assert clff.score(X, y) == 1.0

        # Assert margin obtained is what we asked for
        w = clff.coef_.flatten()
        sv_margin = np.min(np.abs(clff.decision_function(X)/np.linalg.norm(w)))
        
        if np.abs(sv_margin - margin) >= 1e-4:
            print("Prescribed margin %.4f and actual margin %.4f differ (by %.4f)." % (margin, sv_margin, np.abs(sv_margin - margin)))

    else:
        flip_ind = np.random.choice(n, int(n*0.01))
        y[flip_ind] = -y[flip_ind]

    if y01:
        y[y==-1] = 0

    if bias:
        # TODO: Get rid of this later, bias should be handled internally,
        #       this is just for ease of implementation for the Hessian
        X = np.c_[np.ones(n), X]

    return X, y, w, (v0, v1)

