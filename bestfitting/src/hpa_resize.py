
kernel_mode = False

import sys
import cv2
import mlcrate as mlc
import argparse
import os
import numpy as np

from PIL import Image


def do_convert(fname_img):
    img = np.array(Image.open(os.path.join(source_dir, fname_img)),
                   dtype=np.float32)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(dest_dir, fname_img), img)


parser = argparse.ArgumentParser(description='HPA Image Resize')
parser.add_argument('--source', type=str, default="./test", help='source')
parser.add_argument('--dest', type=str, default="./resized", help='dest')
parser.add_argument('--size', type=int, default=512, help='size')
args = parser.parse_args()

if __name__ == '__main__':
    size = args.size
    source_dir = args.source
    dest_dir = args.dest
    n_cpu = 2 if kernel_mode else 4

    os.makedirs(dest_dir, exist_ok=True)
    start_num = max(0, len(os.listdir(dest_dir)) - n_cpu * 2)
    fnames = np.sort(os.listdir(source_dir))[start_num:]
    pool = mlc.SuperPool(n_cpu)
    df_list = pool.map(do_convert, fnames, description='Resizing HPA images')

    print('\nsuccess!')
