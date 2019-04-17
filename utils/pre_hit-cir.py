from lxml import etree

def preprocess_xml_corpus(filenames, fout):
    fout = open(fout, 'w', encoding = 'utf-8')
    for filename in filenames:
        xfile = open(filename)
        text = xfile.read().encode('gbk')
        #解析xml字符串
        xml = etree.XML(text)
        #获取结点
        sent_nodes = xml.xpath('//sent')
        for snode in sent_nodes:
            word_nodes = snode.xpath('word')
            for wnode in word_nodes:
                word = wnode.xpath('@cont')[0].strip()
                pos = wnode.xpath('@pos')[0][0] #只取第一个字母
                #修正abs论西部大开发中的教育优先发展.xml中的问题
                #为什么会把'1 '标注成ws?
                #已通过strip函数解决
                fout.write(word + '/' + pos + '  ')
            fout.write('\n')
        xfile.close()
    fout.close()

folder = r'D:\_files\大三下-中文信息处理\实验资源\HIT-CIR_LTP_Corpora_Sample_v1\HIT_IRLab_LTP_Corpora_Sample\5哈工大信息检索研究室单文档自动文摘语料库'
filenames = [folder + r'\1\应用文\abs讣 告.xml',
    folder + r'\1\应用文\abs中共河北省委关于开除刘青山、张子善党籍的决议.xml',
    folder + r'\1\议论文\abs分析：中国开展登月计划的意义 .xml',
    folder + r'\1\议论文\abs关于学习科技知识.xml',
    folder + r'\1\议论文\abs论西部大开发中的教育优先发展.xml',
    folder + r'\1\议论文\abs正义.xml',
    folder + r'\1\议论文\abs左派和右派.xml']

#preprocess_xml_corpus(filenames, 'test_eva/hit-cir.txt')

filenames = [folder + r'\1\说明文\abs21世纪的早晨穿出蛋白丝之感受.xml']
preprocess_xml_corpus(filenames, 'test_eva/unreg.txt')