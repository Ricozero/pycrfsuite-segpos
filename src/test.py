#coding=utf-8
from sklearn.metrics import classification_report
import os
from collections import Counter
import pycrfsuite

import segword

trainerfile = 'trainer'
testfile = 'test/mytest.txt'

tagger = pycrfsuite.Tagger()
tagger.open(trainerfile)

##### 测试 #####
test_sents = segword.read_seg_file(testfile)
X_test = [segword.sent2features(s) for s in test_sents]
y_test = [segword.sent2labels(s) for s in test_sents]

example_sent = test_sents[0]
print(' '.join(segword.sent2tokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(segword.sent2features(example_sent))))
print("Correct:  ", ' '.join(segword.sent2labels(example_sent)))

##### 评估 #####
y_pred = [tagger.tag(xseq) for xseq in X_test]
print(segword.bmes_classification_report(y_test, y_pred))

##### 输出 #####
segword.write_seg_file(test_sents, testfile + '.r')
print('结果文件已保存为' + testfile + '.r')