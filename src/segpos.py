from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
import numpy as np

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
        test_sents.append([])
        for char in sent:
            if char != ' ' and char != '\n':
                test_sents[num].append(char)
    rawfile.close()
    return test_sents

def read_seg_pos_file(filename, simplified = True):
    '''
    读取已经分词、标注的语料文件

    文件的格式：
        一行表示一句
        每个词后一个'/x'，x表示词性（小写字母），
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
                    postrain_sents[num].append((w[0:k], w[k + 1].lower()))
                else:
                    postrain_sents[num].append((w[0:k], w[k + 1:].lower()))

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
    wfile.close()

def write_seg_pos_file(sents, filename):
    wfile = open(filename, 'w', encoding = 'utf-8')
    for sent in sents:
        for word in sent:
            wfile.write(word[0] + '/' + word[1] + '  ')
        wfile.write('\n')
    wfile.close()

def word2features(sent, i):
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

    trainer = pycrfsuite.Trainer()  #verbose默认开启

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

def viterbi_pred(info, X_test):
    lbdict = info.labels    #标签字典：通过词性查序号
    lblist = []             #标签列表：通过序号查标签
    for lb in lbdict:
        lblist.append(lb)

    tw = info.transitions       #transition weight dictionary
    sw = info.state_features    #state weight ditionary

    y_pred = []
    for sent_feat in X_test:
        VW = np.zeros((len(sent_feat), len(lbdict)))
        for i, word_feat in enumerate(sent_feat):
            for feat in word_feat:
                for lb in lbdict:
                    VW[i][int(lbdict[lb])] += sw.get((feat, lb), 0)
        EW = np.zeros((len(sent_feat), len(lbdict), len(lbdict)))
        for lb1 in lbdict:
            for lb2 in lbdict:
                for i in range(len(sent_feat)):
                    EW[i][int(lbdict[lb1])][int(lbdict[lb2])] = tw.get((lb1, lb2), 0)
        BP = viterbi(VW, EW)
        pos_list = []
        for i in BP:
            pos_list.append(lblist[int(i)])
        y_pred.append(pos_list)
    return y_pred

def viterbi(VW, EW):
    '''
    :param VW:是节点（状态特征）对应的权值，维度表示：序列，标签
    :param EW:是边（转移特征）对应的权值，维度表示：序列，标签1，标签2
    '''
    D = np.full(shape=(np.shape(VW)), fill_value=.0)    #delta
    P = np.full(shape=(np.shape(VW)), fill_value=.0)    #psi
    for i in range(np.shape(VW)[0]):
        #初始化
        if 0 == i:
            D[i] = np.multiply(VW[i], VW[i])
            P[i] = np.zeros(np.shape(VW)[1])
        #递推求解布局最优状态路径
        else:
            for y in range(np.shape(VW)[1]):
                for l in range(np.shape(VW)[1]):
                    #前导状态的最优状态路径的概率 +前导状态到当前状态的转移概率 + 当前状态的概率
                    delta = D[i - 1, l] + EW[i - 1][l, y] + VW[i, y]
                    if 0 == l or delta > D[i, y]:
                        D[i, y], P[i, y] = delta, l
    #返回，得到所有的最优前导状态
    N = np.shape(VW)[0]
    BP = np.full(shape=(N,), fill_value=0.0)
    t_range = -1 * np.array(sorted(-1 * np.arange(N)))
    for t in t_range:
        if N - 1 == t:  #得到最优状态
            BP[t] = np.argmax(D[-1])
        else:           #得到最优前导状态
            BP[t] = P[t + 1, int(BP[t + 1])]

    return BP

if __name__ == '__main__':
    pass