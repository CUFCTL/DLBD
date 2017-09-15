import os, sys
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', dest='root', help='root of your dataset')
parser.add_argument('-y', '--year', dest='year', help='the year for your dataset')

args = parser.parse_args()

if not os.path.exists(args.root):
	print("The path doesn't exist\n")
	sys.exit(1)
	

dest_dir = os.path.join(args.root, 'VOCdevKit')
train_data_f = 'train_data_'+args.year+'.txt'
train_anno_f = 'train_anno_'+args.year+'.txt'
val_data_f = 'val_data_'+args.year+'.txt'
val_anno_f = 'val_anno_'+args.year+'.txt'

imlst  = glob.glob(os.path.join(args.root, 'VOCdevkit', 'VOC'+args.year, 'ImageSets', '*.jpg'))
lablst = glob.glob(os.path.join(args.root, 'VOCdevkit', 'VOC'+args.year, 'labels', '*.txt'))

with open(train_data_f, 'w') as df:
	for im in sorted(imlst):
		f.write("%s\n", im)



