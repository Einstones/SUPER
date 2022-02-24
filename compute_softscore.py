import os
import sys
import json
import _pickle
import numpy as np
sys.path.append(os.getcwd())
sys.path.append('.')
sys.path.append('..')
import utils as utils
import config
from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


def get_file(train=False, val=False, test=False, question=False, answer=False):
    """ Get the correct question or answer file."""
    _file = utils.path_for(train=train, val=val, test=test, question=question, answer=answer)
    with open(_file, 'r') as fd:
        _object = json.load(fd)
    return _object


def get_score(occurences):
    """ Average over all 10 choose 9 sets. """
    score_soft = occurences * 0.3
    score = score_soft if score_soft < 1.0 else 1.0
    return score


def preprocess_answer(answer):
    """ Mimicing the answer pre-processing with evaluation server. """
    dummy_vqa = lambda: None
    dummy_vqa.getQuesIds = lambda: None
    vqa_eval = VQAEval(dummy_vqa, None)
    answer = vqa_eval.processDigitArticle(vqa_eval.processPunctuation(answer))
    answer = answer.replace(',', '')
    return answer


def process_answer(answers_dset):
    occurence = {}
    for ans_entry in answers_dset:
        answers = ans_entry['answers']  # 答案列表
        # 选取次数最多的为gtruth
        ans_cnt = {} # 计数字典
        for each_ans in answers:
            ans = each_ans['answer']
            ans_cnt[ans] = ans_cnt.get(ans, 0) + 1
        # 选取出现次数最多的答案作为gtruth
        top_k = sorted(ans_cnt.items(), key=lambda x: x[1], reverse=True)  # 按照第二个元素的大小进行排序
        gtruth = top_k[0][0] # 取字典最前面的答案作为gtruth
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
    return occurence


def create_ans2label(occurence, name, cache_root):
    """ Map answers to label. """
    label, label2ans, ans2label = 0, [], {}
    for answer in occurence:   # 直接遍历key
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    utils.create_dir(cache_root) # 创建目录

    cache_file = os.path.join(cache_root, name+'_ans2label.pkl')
    _pickle.dump(ans2label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_root, name+'_label2ans.pkl')
    _pickle.dump(label2ans, open(cache_file, 'wb'))
    return ans2label


def compute_target(answers_dset, ans2label, name, cache_root):
    """ Augment answers_dset with soft score as label. """
    target = []
    # 统计数目
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer['answer']
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels, scores = [], []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            score = get_score(answer_count[answer])
            scores.append(score)

        target.append({
            'question_id': ans_entry['question_id'],
            'image_id': ans_entry['image_id'],
            'labels': labels,
            'scores': scores
        })

    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, name+'_target.pkl')
    _pickle.dump(target, open(cache_file, 'wb'))
    return target


if __name__ == '__main__':
    train_answers = get_file(train=True, answer=True)
    val_answers = get_file(val=True, answer=True)
    train_answers = train_answers['annotations']  # 列表
    val_answers = val_answers['annotations']

    answers = train_answers + val_answers
    occurence = process_answer(answers)
    print("create answers to integer labels...")
    ans2label = create_ans2label(occurence, 'trainval', config.cache_root)

    print("converting target for train and val answers...")
    compute_target(train_answers, ans2label, 'train', config.cache_root)
    compute_target(val_answers, ans2label, 'val', config.cache_root)
