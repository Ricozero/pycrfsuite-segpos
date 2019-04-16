#coding=utf-8
from sklearn.metrics import classification_report
import pycrfsuite

import segpos

segmodel = 'models/seg'
posmodel = 'models/pos'
testfile = 'test_eva/hit-cir.txt'
outputfile = 'output/' + testfile[testfile.rfind('/') + 1:]

seg_tagger = pycrfsuite.Tagger()
seg_tagger.open(segmodel)
pos_tagger = pycrfsuite.Tagger()
pos_tagger.open(posmodel)

segtest_sents, postest_sents = segpos.read_seg_pos_file(testfile)
#分词预测
X_test = [segpos.sent2features(s) for s in segtest_sents]
y_test = [segpos.sent2labels(s) for s in segtest_sents]
y_pred = [seg_tagger.tag(xseq) for xseq in X_test]

print('分词结果：')
print(segpos.my_classification_report(y_test, y_pred))

#词性标注预测
#为了正常进行评估，实际上这里的词性标注是基于正确的分词结果
#所以说，分词和词性标注是分割开来的
X_test = [segpos.sent2features(s) for s in postest_sents]
y_test = [segpos.sent2labels(s) for s in postest_sents]
#改用自己的viterbi算法预测
#y_pred = [pos_tagger.tag(xseq) for xseq in X_test]
y_pred = segpos.viterbi_pred(pos_tagger.info(), X_test)

for num, sent in enumerate(postest_sents):
    for i, word in enumerate(sent):
        postest_sents[num][i] = word[0], y_pred[num][i]

print('词性标注结果：')
print(segpos.my_classification_report(y_test, y_pred))

#结果输出
segpos.write_seg_pos_file(postest_sents, outputfile)
print('结果文件已保存为' + outputfile)