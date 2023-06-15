import os
import sys
import torch
from tools import train_net, test_net
from utils import parser
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

def main():

    args = parser.get_args()
    parser.setup(args)    
    print(args)
    if args.test:
        test_net(args)
    else:
        train_net(args)

if __name__ == '__main__':
    main()