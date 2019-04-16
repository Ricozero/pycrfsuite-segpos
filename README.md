# 使用pycrfsuite的自动分词-词性标注一体化算法

改写自[pycrfsuite-segword](https://github.com/Ricozero/pycrfsuite-segword)

## 问题

- 自动标注模型调用```tagger.info()```会报错：AttributeError: 'NoneType' object has no attribute 'group'.  
这是训练集的问题，把训练集中的单空格、三空格改成合乎规则的双空格即可。  详见[github](https://github.com/scrapinghub/python-crfsuite/issues/14).