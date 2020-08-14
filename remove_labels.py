import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("path_input", help="path of input file (TSV, 2 columns, text and label)")
parser.add_argument("path_output", help="path of output file (one text per line, no label)")
args = parser.parse_args()
if os.path.exists(args.path_output):
    raise ValueError("There is already something at '%s'" % args.path_output)

outfile = open(args.path_output, 'w')
with open(args.path_input) as infile:
    for i, line in enumerate(infile):
        elems = line.rstrip().split("\t")
        if len(elems) != 2:
            raise RuntimeError("Expected 2 columns, found %d in line %d" % (len(elems), i+1))
        else:
            text = elems[0]
            label = elems[1]
            outfile.write(text + "\n")
outfile.close()
