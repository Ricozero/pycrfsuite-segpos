#coding=utf-8
import pycrfsuite
import time
import os
from collections import Counter

import segpos

trainfile = 'train/199801.txt'
segmodel = 'models/seg'
posmodel = 'models/pos'

##### 读取训练集 #####
print('开始读取训练集...')
start = time.process_time()
segtrain_sents, postrain_sents = segpos.read_seg_pos_file(trainfile)
end = time.process_time()
print('用时' + str(end - start) + 's')

print('分词训练集示例：')
print(segtrain_sents[0])
print(segpos.sent2features(segtrain_sents[0])[0])
print('词性标注训练集示例：')
print(postrain_sents[0])
print(segpos.sent2features(postrain_sents[0])[0])

##### 分词模型 #####
if os.path.exists(segmodel):
    while 1:
        answer = input('分词训练模型已存在，是否重新训练？（y/n）')
        if answer == 'y' or answer == 'Y':
            do_train_seg = True
            break
        elif answer == 'n' or answer == 'N':
            do_train_seg = False
            break
else:
    do_train_seg = True

if do_train_seg:
    print('开始训练分词模型...')
    start = time.process_time()
    trainer = segpos.train(segtrain_sents, segmodel)
    end = time.process_time()
    print('总用时' + str(end - start) + 's', end = '\n\n')

tagger = pycrfsuite.Tagger()
tagger.open(segmodel)

info = tagger.info()
print("Top likely transitions:")
segpos.print_transitions(Counter(info.transitions).most_common(15))
print("\nTop unlikely transitions:")
segpos.print_transitions(Counter(info.transitions).most_common()[-15:])
print("Top positive:")
segpos.print_state_features(Counter(info.state_features).most_common(20))
print("\nTop negative:")
segpos.print_state_features(Counter(info.state_features).most_common()[-20:])
print('\n')

##### 词性标注模型 #####
if os.path.exists(posmodel):
    while 1:
        answer = input('词性标注训练模型已存在，是否重新训练？（y/n）')
        if answer == 'y' or answer == 'Y':
            do_train_pos = True
            break
        elif answer == 'n' or answer == 'N':
            do_train_pos = False
            break
else:
    do_train_pos = True

if do_train_pos:
    print('开始训练词性标注模型...')
    start = time.process_time()
    trainer = segpos.train(postrain_sents, posmodel)
    end = time.process_time()
    print('总用时' + str(end - start) + 's', end = '\n\n')

tagger = pycrfsuite.Tagger()
tagger.open(posmodel)

info = tagger.info()
print("Top likely transitions:")
segpos.print_transitions(Counter(info.transitions).most_common(15))
print("\nTop unlikely transitions:")
segpos.print_transitions(Counter(info.transitions).most_common()[-15:])
print("Top positive:")
segpos.print_state_features(Counter(info.state_features).most_common(20))
print("\nTop negative:")
segpos.print_state_features(Counter(info.state_features).most_common()[-20:])
