import os
import csv
from collections import Counter
#数据访问
def read_data(dir):
    data = []
    #os.path.join 构建目录并访问
    for category in ['neg','pos']:      
        dir_name = os.path.join(dir,category)
        for filename in os.listdir(dir_name):
            file_path = os.path.join(dir_name,filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                data.append(content)
    return data

#构建词表
def build_vocab(data,vocab_size):
    # 分词并统计单词出现次数
    word_counts = Counter()
    for sentence in data:
        sentence = sentence.replace(".","") #预处理去除句末的句号.
        words = sentence.split()
        word_counts.update(words)
    #方法1
    # 设置vocab的最大容量vocab_size
    sorted_vocab = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:vocab_size])
    return sorted_vocab

    # #方法2
    # # 过滤出现次数不足min_frequence的单词
#     min_frequence = 2
#     filtered_vocab = {word: count for word, count in word_counts.items() if count >= min_frequence}
    # # 按照单词出现次数降序排序
#     sorted_vocab = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))

#保存为文件形式
def save_vocab_to_csv(sorted_vocab, file_path):
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Word', 'Count'])  # 写入表头
        for word, count in sorted_vocab.items():
            csv_writer.writerow([word, count])


train_data = read_data(r'./aclImdb/train') 
test_data = read_data(r'./aclImdb/test')
train_vocab = build_vocab(train_data,1000)
test_vocab = build_vocab(test_data,1000)
save_vocab_to_csv(train_vocab, 'train_vocab.csv')
save_vocab_to_csv(test_vocab, 'test_vocab.csv')
#测试
# print(list(train_vocab.items())[:500])




