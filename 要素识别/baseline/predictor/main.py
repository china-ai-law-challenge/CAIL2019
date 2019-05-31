import json
import jieba
from sklearn.externals import joblib


class Predictor(object):
    def __init__(self, model_dir):
        self.tfidf = joblib.load(model_dir + 'tfidf.model')
        self.tag = joblib.load(model_dir + 'tag.model')
        self.batch_size = 1

        self.cut = jieba

    def predict_tag(self, vec):
        y = self.tag.predict(vec)
        if y[0] == '':
            return []
        return [int(y[0]) + 1]

    def predict(self, content):
        fact = ' '.join(self.cut.cut(str(content)))
        vec = self.tfidf.transform([fact])
        ans = self.predict_tag(vec)
        # print(ans)
        return ans


def generate_pred_file(tags_list, prd, inf_path, outf_path):
    with open(inf_path, 'r', encoding='utf-8') as inf, open(
            outf_path, 'w', encoding='utf-8') as outf:
        for line in inf.readlines():
            pre_doc = json.loads(line)
            predict_doc = []
            for ind in range(len(pre_doc)):
                pred_sent = pre_doc[ind]
                pre_content = pre_doc[ind]['sentence']
                pred_label = prd.predict(pre_content)
                label_names = []
                for label in pred_label:
                    label_names.append(tags_list[label - 1])
                pred_sent['labels'] = label_names
                predict_doc.append(pred_sent)
            json.dump(predict_doc, outf, ensure_ascii=False)
            outf.write('\n')

if __name__ == '__main__':
    input_path_labor = "/input/labor/input.json"
    tag_path_labor = "tags/labor/tags.txt"
    input_path_divorce = "/input/divorce/input.json"
    tag_path_divorce = "tags/divorce/tags.txt"
    input_path_loan = "/input/loan/input.json"
    tag_path_loan = "tags/loan/tags.txt"
    output_path_labor = "/output/labor/output.json"
    output_path_divorce = "/output/divorce/output.json"
    output_path_loan = "/output/loan/output.json"

    # 生成labor领域的预测文件
    print('predict_labor...')
    tags_list = []
    with open(tag_path_labor, 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
    prd = Predictor('model_labor/')
    generate_pred_file(tags_list, prd, input_path_labor, output_path_labor)

    # 生成divorce领域的预测文件
    print('predict_divorce...')
    tags_list = []
    with open(tag_path_divorce, 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
    prd = Predictor('model_divorce/')
    generate_pred_file(tags_list, prd, input_path_divorce, output_path_divorce)

    # 生成loan领域的预测文件
    print('predict_loan...')
    tags_list = []
    with open(tag_path_loan, 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
    prd = Predictor('model_loan/')
    generate_pred_file(tags_list, prd, input_path_loan, output_path_loan)
