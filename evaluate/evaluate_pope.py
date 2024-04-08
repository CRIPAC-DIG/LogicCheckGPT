import json
from tabulate import tabulate


ans_file = './result/logs_mplug_pope/answer_%s_logic_check.json'
label_file = './dataset/pope_coco/coco_50_pope_%s.json'


def evaluate(type: str):
    answers = [json.loads(q) for q in open(ans_file%(type), 'r')]
    label_list = [json.loads(q)['label'] for q in open(label_file%(type), 'r')]

    answers = answers[0:300]
    label_list = label_list
    # print(answers)

    for answer in answers:
        text = answer['answer']

        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words or "don\'t" in words or "cannot" in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    # yes class
    precision_p = float(TP) / float(TP + FP)
    recall_p = float(TP) / float(TP + FN)
    f1_p = 2*precision_p*recall_p / (precision_p + recall_p)
    # no class 
    precision_n = float(TN) / float(TN + FN)
    recall_n = float(TN) / float(TN + FP)
    f1_n = 2*precision_n*recall_n / (precision_n + recall_n)
    acc = (TP + TN) / (TP + TN + FP + FN)

    results = {
                    'Accuracy': acc, 
                    'Precision-P': precision_p, 
                    'Recall-P': recall_p, 
                    'F1 score-P': f1_p, 
                    'Precision-N': precision_n, 
                    'Recall-N': recall_n, 
                    'F1 score-N': f1_n, 
                    'F1 Macro': (f1_p+f1_n)/2,
                    'Yes ratio': yes_ratio
    }

    print("-"*20 + type + "-"*20)
    results2 = {metric: [value] for metric, value in results.items()}
    print(tabulate(results2, headers="keys", tablefmt="fancy_grid", showindex=True, floatfmt=".6f"))



if __name__ == "__main__":
    type = "adversarial"                # "popular","random"
    evaluate(type)