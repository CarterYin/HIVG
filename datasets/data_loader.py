# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
# import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data

sys.path.append('.')


from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.word_utils import Corpus
from pycocotools import mask as coco_mask


def convert_coco_poly_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        # If the mask is empty, it indicates that there is no target. Directly return a mask with a value of 0.
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  # reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


# Bert text encoding

class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class DatasetNotFoundError(Exception):
    pass


class TransVGDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        # TODO: 数据集不一样，全部多了 train_pseudo
        'referit': {'splits': ('train', 'val', 'trainval', 'test', 'train_pseudo')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB', 'train_pseudo'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB', 'train_pseudo'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val', 'train_pseudo'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test', 'train_pseudo'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test', 'train_pseudo')
        },
        'mixup': {
            'splits': ('train', 'val', 'test', 'train_pseudo')
        }
    }

    """ 数据集核心处理部分 """
    def __init__(self, args, data_root, split_root='data', dataset='referit',
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, prompt_template=None, lstm=False,
                 bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.prompt_template = prompt_template
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx = return_idx
        self.use_seg_mask = args.use_seg_mask

        assert self.transform is not None

        if split in ['train', 'train_pseudo']:
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            # TODO: It is flickr30k-images, is not 这里把 flickr30k_images
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        else:  # refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('The dataset {} is not found!'.format(osp.join(self.split_root, self.dataset)))
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

        # if self.prompt_template:
        #     self.images = self.prompt(self.images)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr':  # flickr
            img_file, bbox, phrase = self.images[idx]
            img_size = None
            obj_mask = None
            image_size = []
            bbox_xywh = bbox.copy()
        else:
            img_file, img_size, bbox, phrase, obj_mask = self.images[idx]  # The most original data
            bbox_xywh = bbox.copy()
            if isinstance(img_size, dict):
                image_size = [img_size["height"], img_size["width"]]
            # else:
            #     img_size = None

        ## box format: to x1y1x2y2
        bbox_ori = bbox.copy()
        #  For the refcoco dataset, uniformly convert the bbox from xywh (horizontal x, vertical y) to x1y1x2y2
        #  format, and then convert it back to xywh at normalizationandpad in Transform.
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)
            bbox_xywh[2], bbox_xywh[3] = bbox_xywh[2] - bbox_xywh[0], bbox_xywh[3] - bbox_xywh[1]

        #  Loading grounding pre-training data for mixed datasets
        if self.dataset == "mixup":
            if img_file.split("_")[0] == "COCO":
                dataset = "coco"
                im_dir = osp.join(self.data_root, 'other', 'images', 'mscoco', 'images', 'train2014')
            else:
                if img_size == "flickr":
                    dataset = "flickr"
                    im_dir = osp.join(self.data_root, 'Flickr30k', 'flickr30k-images')
                elif img_size == "referit":
                    dataset = "referit"
                    im_dir = osp.join(self.data_root, 'referit', 'images')
                else:
                    print("img_file：", img_file, 'img_size: ', img_size, "bbox: ", bbox, "phrases: ", phrase)
                    raise ValueError('Can not find image dir')
        else:
            dataset = self.dataset
            im_dir = self.im_dir

        img_path = osp.join(im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        if dataset == 'referit' or dataset == 'flickr':
            image_size = [img.height, img.width]

        bbox = torch.tensor(bbox)
        bbox = bbox.float()

        if dataset in ["unc", "unc+", "gref", "gref_umd", "coco"]:
            h, w = image_size[0], image_size[1]
            if self.use_seg_mask:
                bool_obj_mask = convert_coco_poly_mask([obj_mask], h, w)  # torch.Size([1, 480, 640])
            else:  # use box mask supervised
                obj_mask = [bbox_xywh]  # Here, the bbox is required to be in the xywh format.
                bool_obj_mask = convert_coco_poly_mask(np.array([obj_mask]), h, w)  # torch.Size([1, 480, 640])
        else:
            h, w = image_size[0], image_size[1]
            obj_mask = [list(map(float, bbox_xywh))]
            bool_obj_mask = convert_coco_poly_mask(np.array([obj_mask]), h, w)  # torch.Size([1, 480, 640])

        return img_file, img, phrase, bbox, bbox_ori, bool_obj_mask

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def prompt(self, sample_list):
        n = len(sample_list)
        new_sample_list = []

        for i in range(n):
            if self.dataset == 'flickr':
                tmp_sample = (sample_list[i][0], sample_list[i][1], self.prompt_template.replace('{pseudo_query}', sample_list[i][2]))
            else:
                tmp_sample = (sample_list[i][0], sample_list[i][1], sample_list[i][2],
                              self.prompt_template.replace('{pseudo_query}', sample_list[i][3]), sample_list[i][4])
            new_sample_list.append(tmp_sample)
        return new_sample_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file, img, phrase, bbox, bbox_ori, obj_mask = self.pull_item(idx)
        phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase, 'obj_mask': obj_mask}
        input_dict = self.transform(input_dict)

        img = input_dict['img']
        img_mask = input_dict['mask']
        bbox = input_dict['box']
        phrase = input_dict['text']
        obj_mask = input_dict['obj_mask']

        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id > 0, dtype=int)
        else:
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask

        text = []
        text_mask = []

        """ # old code
        if self.testmode:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                   np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            # print(img.shape)
            return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)
        """

        if self.testmode:
            return img, np.array(text, dtype=int), np.array(text_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                   np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:  # Avoid 7 return values
            return img, np.array(img_mask), np.array(text, dtype=int), np.array(text_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), img_file, phrase, bbox_ori, np.array(obj_mask, dtype=int)

    def getitem_for_origin_transvg(self, idx):
        img_file, img, phrase, bbox, bbox_ori = self.pull_item(idx)

        phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']
        img_mask = input_dict['mask']

        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id > 0, dtype=int)
        else:
            # encode phrase to bert input
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask

        if self.testmode:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                   np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32)


