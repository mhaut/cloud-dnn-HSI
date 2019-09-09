# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import scipy.io as sio
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path of image")
parser.add_argument("height", type=int, help="Image height")
parser.add_argument("width", type=int, help="Image width")
parser.add_argument("--channels_last", dest = 'channels_last', default=False, action='store_true', help='Set this flags is image is (Height, Width, Channels)')
parser.add_arument("--labels_file", dest = 'labels_file', default=None)
parser.add_argument("--output_file", dest = 'output', default="output.csv", help='Set output file name')
parser.add_argument("--is_matlab_file", dest = 'matlab', default=False, action='store_true')
parser.add_argument("--dict_key", dest = 'key', default=None)
parser.add_argument("--dict_key_gt", dest = 'key_gt', default=None)

args = parser.parse_args()
gt = None

if args.matlab:
    im = sio.loadmat(args.image)[args.key]
    if args.labels_file != None:
        gt = sio.loadmat(args.labels_file)[args.key_gt]
        gt = gt.reshape(args.height*args.width, 1)


else:
    im = io.imread(args.image)
    
array = np.array(im, dtype=np.uint16)
if args.channels_last:
    array = array.reshape(args.height*args.width, array.shape[-1])
    if gt != None:
        array = np.hstack((gt, array))
else:
    array = array.reshape((array.shape[0], args.height*args.width))
    array = np.transpose(array)
    if gt != None:
        array = np.hstack((gt, array))

with open(args.output, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in array:
            writer.writerow(line)