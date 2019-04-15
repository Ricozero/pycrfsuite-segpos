#coding=utf-8
from sklearn.metrics import classification_report
import pycrfsuite
import time
import os
from collections import Counter

import segpos

#训练模型文件
trainerfile = 'trainer'

#用msr训练，tagger.info()会报错
#trainfile = 'train/msr_training.utf8'
#testfile = 'test/msr_test_gold.utf8'
trainfile = 'train/pku_training.utf8'
testfile = 'test/pku_test_gold.utf8'

##### 训练 #####
print('开始读取训练集...')
start = time.process_time()
train_sents = segpos.read_seg_file(trainfile)
end = time.process_time()
print('用时' + str(end - start) + 's')
#print('\n')

print(train_sents[0])
print(segpos.sent2features(train_sents[0])[0])

#判断训练文件是否存在，存在则手动决定是否训练
if os.path.exists(trainerfile):
    while 1:
        answer = input('训练模型已存在，是否重新训练？（y/n）')
        if answer == 'y' or answer == 'Y':
            do = True
            break
        elif answer == 'n' or answer == 'N':
            do = False
            break
else:
    do = True

if do:
    print('开始训练...')
    start = time.process_time()
    trainer = segpos.train(train_sents)
    end = time.process_time()
    print('用时' + str(end - start) + 's')
    print(len(trainer.logparser.iterations))
    print(trainer.logparser.last_iteration)

tagger = pycrfsuite.Tagger()
tagger.open(trainerfile)

##### 测试 #####
test_sents = segpos.read_seg_file(testfile)
X_test = [segpos.sent2features(s) for s in test_sents]
y_test = [segpos.sent2labels(s) for s in test_sents]

example_sent = test_sents[0]
print(' '.join(segpos.sent2tokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(segpos.sent2features(example_sent))))
print("Correct:  ", ' '.join(segpos.sent2labels(example_sent)))

##### 评估 #####
y_pred = [tagger.tag(xseq) for xseq in X_test]
print(segpos.bmes_classification_report(y_test, y_pred))

info = tagger.info()

print("Top likely transitions:")
segpos.print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
segpos.print_transitions(Counter(info.transitions).most_common()[-15:])

print("Top positive:")
segpos.print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
segpos.print_state_features(Counter(info.state_features).most_common()[-20:])

##### 输出 #####
segpos.write_seg_file(test_sents, testfile + '.r')
print('结果文件已保存为' + testfile + '.r')