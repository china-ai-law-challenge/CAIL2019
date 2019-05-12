def get_score(input_path, ground_truth_path, output_path):
    cnt = 0
    f1 = open(ground_truth_path,"r")
    f2 = open(output_path,"r")
    correct = 0
    for line in f1:
        cnt += 1
        try:
            if line[:-1] == f2.readline()[:-1]:
                correct += 1
        except:
            pass

    return 1.0*correct/cnt
