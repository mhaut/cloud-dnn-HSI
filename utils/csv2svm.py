#!/usr/bin/env python

import sys
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path of image input")
    parser.add_argument("output", help="Output path")
    parser.add_argument("--label_column", type=int, default=0)
    parser.add_argument("--no_labels", dest = 'no_labels', default=False, action='store_true', help='Set this flag if your csv file have no labels')
    
    args = parser.parse_args()
    return args

def construct_line( label, line ):
	new_line = []
	if float( label ) == 0.0:
		label = "0"
	new_line.append( label )

	for i, item in enumerate( line ):
		if item == '' or float( item ) == 0.0:
			continue
		new_item = "%s:%s" % ( i + 1, item )
		new_line.append( new_item )
	new_line = " ".join( new_line )
	new_line += "\n"
	return new_line

# ---

args = parse_args()

input_file = args.input
output_file = args.output

i = open( input_file, 'rb' )
o = open( output_file, 'wb' )

reader = csv.reader( i )

for line in reader:
	if args.no_labels:
		label = '1'
	else:
		label = line.pop(args.label_column)

	new_line = construct_line( label, line )
	o.write( new_line )
