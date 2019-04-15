#coding=utf-8
from sklearn.metrics import classification_report
import pycrfsuite

import segpos

segmodel = 'models/seg'
posmodel = 'models/pos'
testfile = 'test_eva/test_eva.txt'
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
X_test = [segpos.sent2features(s) for s in postest_sents]
y_test = [segpos.sent2labels(s) for s in postest_sents]
y_pred = [pos_tagger.tag(xseq) for xseq in X_test]

print('词性标注结果：')
print(segpos.my_classification_report(y_test, y_pred))

#结果输出
segpos.write_seg_pos_file(postest_sents, outputfile)
print('结果文件已保存为' + outputfile)