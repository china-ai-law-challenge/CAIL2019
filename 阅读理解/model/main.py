#coding: utf-8
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser('test!!!')
    parser.add_argument('--infile', dest="data_file", help='Input data JSON file.')
    parser.add_argument('--outfile', dest="result_file", help='Output result Json file.')
    
    return parser.parse_args()

def main(args):
    
    data = open(args.infile,"r")
    result = open(args.outfile,"w")

if __name__ == '__main__':
    args = parse_args()
    main(args)
