import json

input_path_labor = "/input/labor/input.json"
input_path_divorce = "/input/divorce/input.json"
input_path_loan = "/input/loan/input.json"
output_path_labor = "/output/labor/output.json"
output_path_divorce = "/output/divorce/output.json"
output_path_loan = "/output/loan/output.json"


def predict(input_path, output_path):
    inf = open(input_path, "r", encoding='utf-8')
    ouf = open(output_path, "w", encoding='utf-8')

    for line in inf:
        pre_doc = json.loads(line)
        new_pre_doc = []
        for sent in pre_doc:
            sent['labels'] = []  # 将该空列表替换成你的模型预测的要素列表结果
            new_pre_doc.append(sent)
        json.dump(new_pre_doc, ouf, ensure_ascii=False)
        ouf.write('\n')

    inf.close()
    ouf.close()


# labor领域预测
predict(input_path_labor, output_path_labor)

# loan领域预测
predict(input_path_loan, output_path_loan)

# divorce领域预测
predict(input_path_divorce, output_path_divorce)
