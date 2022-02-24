import os
import json
import h5py
import torch
import _pickle

import numpy as np
import itertools
import utils
import config 
from torch.utils.data import Dataset
torch.utils.data.ConcatDataset.__getattr__ = lambda self, attr: getattr(self.datasets[0], attr)


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        # print(self.word2idx['how'])
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(
            ',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        # print(words)
        tokens = []
        if add_word:

            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        _pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to {}'.format(path))

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from {}'.format(path))
        word2idx, idx2word = _pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(cache_path, name, img_id2val):
    """ Load entries. img_id2val: dict {img_id -> val} ,
        val can be used to retrieve image or features.
    """
    train, val, test = False, False, False
    if name == 'train':
        train = True
    elif name == 'val':
        val = True
    else:
        test = True
    question_path = utils.path_for(train=train, val=val, test=test, question=True)
    questions = json.load(open(question_path))
    if not config.cp_data:
        questions = questions['questions']
    questions = sorted(questions, key=lambda x: x['question_id'])
    if test: # will be ignored anyway
        answers = [
            {'image_id': 0, 'question_id': 0,
            'labels': [], 'scores': []} 
            for _ in range(len(questions))]
    else:
        answer_path = os.path.join(cache_path, '{}_target.pkl'.format(name))
        answers = _pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])
        utils.assert_eq(len(questions), len(answers))

    entries = []
    for question, answer in zip(questions, answers):
        if not test:
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test']
        self.dictionary = dictionary

        # loading answer-label
        self.ans2label = _pickle.load(open(os.path.join(
            config.cache_root, 'trainval_ans2label.pkl'), 'rb'))
        self.label2ans = _pickle.load(open(os.path.join(
            config.cache_root, 'trainval_label2ans.pkl'), 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        # loading image features
        image_split = 'test' if name == 'test' else 'trainval'
        self.img_id2idx = _pickle.load(open(os.path.join(config.ids_path, 'okvqa_{0}36_imgid2idx.pkl'.format(image_split)), 'rb'))
        self.h5_path = os.path.join(config.bottom_up_path, 'okvqa_{}2014_36fixed_features.h5'.format(image_split))
        if config.in_memory:
            print('loading image features from h5 file')
            with h5py.File(self.h5_path, 'r') as hf:
                self.features = np.array(hf.get('image_features'))
                self.spatials = np.array(hf.get('spatial_features'))

        self.entries = _load_dataset(config.cache_root, name, self.img_id2idx)

        self.tokenize()
        self.tensorize()
        self.v_dim = config.output_features
        self.s_dim = config.num_fixed_boxes

    def tokenize(self, max_length=config.max_question_len):
        """ Tokenizes the questions.
            This will add q_token in each entry of the dataset.
            -1 represent nil, and should be treated as padding_idx in embedding.
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        if config.in_memory:
            self.features = torch.from_numpy(self.features)
            self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None


    def load_image(self, image_id):
        """ Load one image feature. """
        # open h5 file
        self.image_feat = h5py.File(self.h5_path, 'r')
        # convert image_id into idx
        idx = self.img_id2idx[image_id]  # int -> int
        features = self.image_feat['features'][idx]  # (36, 2048)
        bboxes = self.image_feat['boxes'][idx]  # (36, 4)
        obj_cls = self.image_feat['objects_label'][idx]  # (36,)
        attr_cls = self.image_feat['attrs_label'][idx] # (36, )
        return torch.from_numpy(features), torch.from_numpy(bboxes), torch.from_numpy(obj_cls).long(), torch.from_numpy(attr_cls).long()

    def __getitem__(self, index):
        entry = self.entries[index]
        if config.in_memory:
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]
        else:
            features, spatials, object_cls, attr_cls = self.load_image(entry['image_id'])

        question_id = entry['question_id']
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, spatials, object_cls, attr_cls, question, target, question_id

    def __len__(self):
        return len(self.entries)


def tfidf_from_questions(names, dictionary, dataroot='data', target=['vqa', 'vg', 'cap', 'flickr']):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)

    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0]); inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1]); inds[1].append(c[0])

    if 'vqa' in target:  # VQA 2.0
        for name in names:
            assert name in ['train', 'val', 'test-dev2015', 'test2015']
            question_path = os.path.join(
                dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' % \
                (name + '2014' if 'test'!=name[:4] else name))
            questions = json.load(open(question_path))['questions']

            for question in questions:
                populate(inds, df, question['question'])

    if 'vg' in target:  # Visual Genome
        question_path = os.path.join(dataroot, 'question_answers.json')
        vgq = json.load(open(question_path, 'r'))
        for vg in vgq:
            for q in vg['qas']:
                populate(inds, df, q['question'])

    if 'cap' in target: # MSCOCO Caption
        for split in ['train2017', 'val2017']:
            captions = json.load(open('data/annotations/captions_%s.json' % split, 'r'))
            for caps in captions['annotations']:
                populate(inds, df, caps['caption'])

    # TF-IDF
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))
    return tfidf, weights
