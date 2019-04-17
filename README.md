# 使用pycrfsuite的自动分词-词性标注一体化算法

中文信息处理第二次作业  
分词部分改写自第一次作业：[pycrfsuite-segword](https://github.com/Ricozero/pycrfsuite-segword)  
segpos = word segmentation + Part of Speech tagging

实验环境：

- Windows 10
- Python 3.7

使用的Python包：

- collections
- scikit-learn
- pycrfsuite
- numpy
- collections
- lxml

---

## 训练阶段

运行`run_train.bat`进行训练，  
训练的语料库由`train.py`的`trainfile`变量决定。  
训练成功后，models文件夹里面会多出`seg`和`pos`两个模型文件，分别对应分词模型和词性标注模型。  
词性标注的训练默认是简化的，即只取第一个字母，可以通过`read_seg_pos_file`函数的simplified参数调整。

### 语料库要求：

一行一句，句子中每个词后跟着'/x'和两个空格，x表示词性。

---

## 测试阶段

### 生语料测试

运行`run_test_raw.bat`，  
测试的文本文件由`test_raw.py`的`testfile`变量决定。  
测试成功后，测试文件的同名文件被输出到output目录。  

过程：先进行分词预测，再根据预测结果进行词性标注。

### 熟语料测试

运行`run_test_eva.bat`，  
测试的文本文件由`test_eva.py`的`testfile`变量决定。  
测试成功后，输出结果评估，测试文件的同名文件被输出到output目录。  

过程：先进行分词预测，对预测结果进行评估；然后使用熟语料的分词进行词性标注预测并评估。  
这样做，分词和词性标注实际上是割裂开的，但是便于进行评估。

### 预测算法

用了两种预测算法，分词是用pycrfsuite的，词性标注是用自己的。

    #调用pycrfsuite的预测算法：
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    #调用自己的viterbi预测算法：
    y_pred = segpos.viterbi_pred(tagger.info(), X_test)

经过测试，两个算法的准确率是相同的，估计crfsuite同样是用viterbi算法进行预测的。

---

## 文件结构

注1：*斜体*表示文件是程序运行后生成的  
注2：所有处理后的文本的编码格式都是utf-8

- models
  - *训练出来的模型（pos和seg）*
- output
  - *分词、词性标注的预测结果*
- src
  - `segpos.py`:分词与词性标注的核心算法与数据处理
  - `test_eva.py`:包含评估的预测，需要有熟语料
  - `test_raw.py`:对原始语料进行预测
  - `train.py`:从语料训练分词和词性标注模型
- test_eva
  - 熟语料文本文件
  - `hit-cir.txt`:来自哈工大信息检索研究中心
  - `ambiguity.txt`:测试歧义
  - `unreg.txt`:测试未登录词
- test_raw
  - 生语料文本文件
- train
  - 用于训练的熟语料文件
  - `199801.txt`:经过预处理的人民日报语料库
- utils
  - `pre_199801.py`:用于预处理人民日报语料库
  - `pre_hit-cir.py`:用于预处理哈工大信息检索研究中心的语料库

---

## 问题

- 人民日报语料库训练出来的自动标注模型调用`tagger.info()`会报错：AttributeError: 'NoneType' object has no attribute 'group'.  
直接原因是存在空标签。其实是训练集的问题，把训练集中的单空格、三空格改成合乎规则的双空格即可。
详见[github](https://github.com/scrapinghub/python-crfsuite/issues/14).