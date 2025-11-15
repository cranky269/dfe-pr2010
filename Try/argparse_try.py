import argparse

# 创建解析器 
parser = argparse.ArgumentParser(description="示例：位置参数的使用。")

# 定义两个位置参数
parser.add_argument('first', type=int, help="第一个整数") 
parser.add_argument('second', type=str, help="第二个字符串")

# 解析命令行参数 
args = parser.parse_args()

# 输出参数值 
print(args.first)
print(args.second) 
