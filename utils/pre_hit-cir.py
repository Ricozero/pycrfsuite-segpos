from lxml import etree

xfile = open(r'D:\_files\大三下-中文信息处理\实验资源\HIT-CIR_LTP_Corpora_Sample_v1\HIT_IRLab_LTP_Corpora_Sample\5哈工大信息检索研究室单文档自动文摘语料库\1\应用文\abs讣 告.xml')
text = xfile.read().encode('gbk')

#解析xml字符串
xml = etree.XML(text)
#获取结点
sent_nodes = xml.xpath('//sent')