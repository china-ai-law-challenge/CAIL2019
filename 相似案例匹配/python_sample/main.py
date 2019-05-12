import random

input_path = "/input/input.txt"
output_path = "/output/output.txt"

inf = open(input_path,"r")
ouf = open(output_path,"w")

for line in inf:
    print("d"+str(random.randint(1,2)),file=ouf)

inf.close()
ouf.close()
