import random

input_path = "/input/input.txt"
output_path = "/output/output.txt"

inf = open(input_path,"r",encoding="utf8")
ouf = open(output_path,"w",encoding="utf8")

for line in inf:
    print(["B","C"][random.randint(0,1)],file=ouf)

inf.close()
ouf.close()
