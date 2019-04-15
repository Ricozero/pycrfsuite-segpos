from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite

def read_raw_file(filename):
    '''
    读取未经过分词、词性标注的测试文件

    文件格式：
        一行表示一句
    '''
    rawfile = open(filename, encoding = 'utf-8')
    sents = rawfile.readlines()

    test_sents = []
    for num, sent in enumerate(sents):
        if sent == '\n':
            continue

        test_sents.append([])
        for char in sent:
            if char != ' ' and char != '\n':
                test_sents[num].append(char)
    return test_sents

def read_seg_pos_file(filename, simplified = True):
    '''
    读取已经分词、标注的语料文件

    文件的格式：
        一行表示一句
        每个词后一个'/x'，x表示词性，
        然后是两个空格

    filename: 已分词、词性标注的语料文件名
    simplified: 词性只取第一个字母
    '''
    spfile = open(filename, encoding = 'utf-8')
    sents = spfile.readlines()
    
    segtrain_sents = []
    postrain_sents = []

    for num, sent in enumerate(sents):
        if sent == '\n':
            continue

        segtrain_sents.append([])
        postrain_sents.append([])

        pre = 0 #当前词的开始
        for cur, char in enumerate(sent):
            if char == ' ' and sent[cur - 1] != ' ':
                w = sent[pre:cur]
                k = w.rfind('/')
                #分词训练集
                if k == 1:
                    segtrain_sents[num].append((w[0], 'S'))
                else:
                    segtrain_sents[num].append((w[0], 'B'))
                    for i in range(1, k - 1):
                        segtrain_sents[num].append((w[0], 'M'))
                    segtrain_sents[num].append((w[k - 1], 'E'))
                #词性标注训练集
                if simplified:
                    postrain_sents[num].append((w[0:k], w[k + 1]))
                else:
                    postrain_sents[num].append((w[0:k], w[k + 1:]))

                pre = cur + 2
            else:
                continue
    spfile.close()
    return segtrain_sents, postrain_sents

def write_seg_file(sents, filename):
    wfile = open(filename, 'w', encoding = 'utf-8')
    for sent in sents:
        for c in sent:
            if c[1] == 'S':
                wfile.write(c[0] + '  ')
            elif c[1] =='B' or c[1] == 'M':
                wfile.write(c[0])
            elif c[1] == 'E':
                wfile.write(c[0] + '  ')
            else:
                print('Invalid tuple: ', end = '')
                print(c)
        wfile.write('\n')

def write_seg_pos_file(sents, filename):
    wfile = open(filename, 'w', encoding = 'utf-8')
    for sent in sents:
        for word in sent:
            wfile.write(word[0] + '/' + word[1] + '  ')
        wfile.write('\n')

def word2features(sent, i):
    '''
    pku训练集+测试集
                    precision    recall  f1-score   support

                B       0.94      0.96      0.95     56883
                E       0.94      0.96      0.95     56883
                M       0.84      0.78      0.81     11479
                S       0.95      0.92      0.93     47489

    micro avg           0.94      0.94      0.94    172734
    macro avg           0.92      0.91      0.91    172734
    weighted avg        0.94      0.94      0.94    172734
    samples avg         0.94      0.94      0.94    172734

    pku训练集+测试集（无-1,-2,+1,+2）
                    precision    recall  f1-score   support

                B       0.93      0.96      0.94     56883
                E       0.93      0.96      0.95     56883
                M       0.84      0.78      0.81     11479
                S       0.95      0.91      0.93     47489

    micro     avg       0.93      0.93      0.93    172734
    macro     avg       0.91      0.90      0.91    172734
    weighted  avg       0.93      0.93      0.93    172734
    samples   avg       0.93      0.93      0.93    172734
    '''
    word = sent[i][0]
    features = [
        'bias',
        word
    ]
    if i > 0:
        wordm1 = sent[i-1][0]
        features.extend([
             '-1:' + wordm1 + word
        ])
        if i > 1:
            wordm2 = sent[i-2][0]
            features.extend([
                '-2:' + wordm2 + wordm1 + word
            ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        wordp1 = sent[i+1][0]
        features.extend([
            '+1:' + word + wordp1
        ])
        if i < len(sent)-2:
            wordp2 = sent[i+2][0]
            features.extend([
                '+2:' + word + wordp1 + wordp2
            ])
    else:
        features.append('EOS')

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

def train(sents, filename):
    X_train = [sent2features(s) for s in sents]
    y_train = [sent2labels(s) for s in sents]

    trainer = pycrfsuite.Trainer(verbose=False)

    #zip函数可以使得X和y的每一个元素按顺序组成元组
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    #trainer.params()
    trainer.train(filename)
    return trainer

def my_classification_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_)
    tagset = sorted(tagset)
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))

if __name__ == '__main__':
    pass