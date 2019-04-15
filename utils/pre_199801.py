#对原始的人民日报语料预处理，使其符合格式
#去掉开始的日期以及分隔文章的空行
#将编码修改为utf-8
fin = open('train/199801.raw', encoding = 'gbk')
fout = open('train/199801.txt', 'w', encoding = 'utf-8')
lines = fin.readlines()
for l in lines:
    fout.write(l[23:])
fin.close()
fout.close()