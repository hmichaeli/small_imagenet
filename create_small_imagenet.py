import os
import torch
from torchvision import datasets, transforms as T
from tqdm import tqdm
import argparse
import shutil
from os import listdir
from os.path import isfile, join

DATA_PATH='/media/ssd/ehoffer/imagenet/'

def imagenet_stats(data_path):
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    means = []
    stds = []
    for dir in ['train', 'val', 'test']:
        root = os.path.join(data_path, dir)
        print("load dir: ", dir)
        dataset = datasets.ImageFolder(root, transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=50,
            num_workers=8,
            drop_last=False,
        )

        for i, img in enumerate(tqdm(loader)):
            means.append(torch.mean(img[0], dim=(-1,-2)))
            stds.append(torch.std(img[0], dim=(-1,-2)))

    mean = torch.mean(torch.cat(means), dim=0)
    std = torch.mean(torch.cat(stds), dim=0)

    print(mean)
    print(std)



def make_tiny_imagenet(wnids, out_dir, imagenet_dir, train_num=500, test_num=50):
    if os.path.isdir(out_dir):
        print('Output directory already exists')
        return

    os.mkdir(out_dir)
    for dir in ['train', 'val', 'test']:
        os.mkdir(os.path.join(out_dir, dir))

    for i, wnid in enumerate(wnids):
        # copy val
        in_val = os.path.join(imagenet_dir, 'val/' + wnid)
        out_val = os.path.join(out_dir, 'val/' + wnid)
        print("copy {} to {}".format(in_val, out_val))
        shutil.copytree(in_val, out_val)

        # copy train
        in_train = os.path.join(imagenet_dir, 'train/' + wnid)
        out_train = os.path.join(out_dir, 'train/' + wnid)
        out_test = os.path.join(out_dir, 'test/' + wnid)
        os.mkdir(out_train)
        os.mkdir(out_test)
        train_files = [f for f in listdir(in_train) if isfile(join(in_train, f))]

        for f in train_files[:train_num]:
            in_f = os.path.join(in_train, f)
            out_f = os.path.join(out_train, f)
            shutil.copyfile(in_f, out_f)

        for f in train_files[train_num: train_num + test_num]:
            in_f = os.path.join(in_train, f)
            out_f = os.path.join(out_test, f)
            shutil.copyfile(in_f, out_f)




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wnid_file', type=argparse.FileType('r'),
                        default='/home/hagaymi/cnn_project2/small_imagenet/200_wnids.txt')
    # parser.add_argument('--num_train', type=int, default=100)
    # parser.add_argument('--num_val', type=int, default=100)
    # parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--out_dir', type=str, default='/home/hagaymi/data/imagenet200')
    parser.add_argument('--imagenet_dir', type=str, default='/media/ssd/ehoffer/imagenet/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    wnids = [line.strip() for line in args.wnid_file]
    make_tiny_imagenet(wnids=wnids, imagenet_dir=args.imagenet_dir, out_dir=args.out_dir)
    imagenet_stats(args.out_dir)
    print()