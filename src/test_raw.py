#coding=utf-8
from sklearn.metrics import classification_report
import pycrfsuite

import segpos

segmodel = 'models/seg'
posmodel = 'models/pos'
testfile = 'test_raw/test_raw.txt'
outputfile = 'output/' + testfile[testfile.rfind('/') + 1:]

seg_tagger = pycrfsuite.Tagger()
seg_tagger.open(segmodel)
pos_tagger = pycrfsuite.Tagger()
pos_tagger.open(posmodel)

#分词预测
segtest_sents = segpos.read_raw_file(testfile)
X_test = [segpos.sent2features(s) for s in segtest_sents]
y_pred = [seg_tagger.tag(xseq) for xseq in X_test]

#词性标注预处理
postest_sents = []
for num, sent in enumerate(segtest_sents):
    postest_sents.append([])
    w = ''
    for i, char in enumerate(sent):
        if y_pred[num][i] == 'S':
            postest_sents[num].append([char])
        elif y_pred[num][i] == 'B' or y_pred[num][i] == 'M':
            w = w + char
        else:
            postest_sents[num].append([w + char])
            w = ''

#词性标注预测
X_test = [segpos.sent2features(s) for s in postest_sents]
#改用自己的viterbi算法预测
#y_pred = [pos_tagger.tag(xseq) for xseq in X_test]
y_pred = segpos.viterbi_pred(pos_tagger.info(), X_test)
#将预测结果加入句子列表
for num, sent in enumerate(postest_sents):
    for i, word in enumerate(sent):
        word.append(y_pred[num][i])

#结果输出
segpos.write_seg_pos_file(postest_sents, outputfile)
print('结果文件已保存为' + outputfile)