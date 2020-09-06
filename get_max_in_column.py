""" Get maximum value in column of a TSV file. """

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path_tsv", type=str)
parser.add_argument("col_index", type=int)
args = parser.parse_args()

# Load data
with open(args.path_tsv) as f:
    data = []
    for line in f:
        elems = line.strip().split("\t")
        data.append(elems[args.col_index])

col_name = data[0]
data = data[1:]
data = [float(x) for x in data]
print("Col name: %s" % col_name)
print("First value: %f" % data[0])
print("Last value: %f" % data[-1])
print("Max value: %f" % max(data))
