# python 解析命令行参数

# 如果脚本很简单或临时使用，没有多个复杂的参数选项，可以直接利用sys.argv将脚本后的参数依次读取(读进来的默认是字符串格式)。
# import sys
# print("输入的参数为:%s" % sys.argv[1])

# 利用argparse模块

import argparse
parser=argparse.ArgumentParser(description="A description of what the program does")
parser.add_argument('--toy','-t',action='store_true',help='Use only 50K samples of data')
parser.add_argument('--num_epochs',choices=[5,10,20],default=5,type=int,help='Number of epochs.')
parser.add_argument("--num_layers", type=int, required=True, help="Network depth.")

args=parser.parse_args()
print(args)
print(args.toy,args.num_epochs,args.num_layers)