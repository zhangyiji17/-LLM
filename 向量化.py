import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Flatten

class WordEmbedding:
    def __init__(self, self_path):
        self.word_to_index, self.word_to_vec_map, self.index_to_word = self.load_embeddings(self_path)

    def load_embeddings(self, self_path):
        with open(self_path, 'r', encoding='utf8') as f:
            words = set()  # 用来存储所有的单词
            word_to_vec_map = {}  # 用来存储单词到词向量的映射
            index = 1
            for line in f:
                line = line.strip().split()  # 按空格分割每一行
                if index == 1:
                    # print(line)  # 打印第一行
                    # print(len(line))  # 打印第一行的长度
                    index += 1
                curr_word = line[0]  # 获取当前单词
                words.add(curr_word)  # 将当前单词添加到单词集合中
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)  # 将当前单词和对应的词向量添加到映射中

            i = 1
            words_to_index = {}  # 用来存储单词到索引的映射
            index_to_words = {}  # 用来存储索引到单词的映射
            for w in sorted(words):
                words_to_index[w] = i  # 将单词和索引添加到映射中
                index_to_words[i] = w  # 将索引和单词添加到映射中
                i += 1
        return words_to_index, word_to_vec_map, index_to_words

    def test(self, word1, word2):
        if word1 not in self.word_to_vec_map or word2 not in self.word_to_vec_map:
            return None
        embedding1 = self.word_to_vec_map[word1]
        embedding2 = self.word_to_vec_map[word2]
        # 计算点积
        dot_product = np.dot(embedding1, embedding2)
        # 计算模
        embedding1_norm = np.linalg.norm(embedding1)
        embedding2_norm = np.linalg.norm(embedding2)
        if embedding1_norm == 0 or embedding2_norm == 0:
            return None
        # 计算余弦相似度
        cosine_similarity = dot_product / (embedding1_norm * embedding2_norm)
        return cosine_similarity


file_path = 'D:\RAG与LLM原理及实践\dataset\glove.6B\glove.6B.50d.txt'
word_embedding = WordEmbedding(file_path)
print('father-mother', word_embedding.test('father', 'mother'))
print('banana-mother', word_embedding.test('banana', 'mother'))

# 示例文本
texts = ['张某是一位智能编程助手', '这是测试文本']
tokenizers_texts = [['张某', '是', '一位', '智能', '编程', '助手'], ['这', '是', '测试', '文本']]

# 将分词结果转换为空格分隔的字符串，便于Tokenizer处理
texts_for_tokenizer = [' '.join(token) for token in tokenizers_texts]

# 使用Tokenizer对文本进行分词
tokenizer = Tokenizer(num_words=1000)  # 假设词汇表大小不超过1000
tokenizer.fit_on_texts(texts_for_tokenizer)  # 训练Tokenizer
word_index = tokenizer.word_index  # 获取单词到索引的映射
sequences = tokenizer.texts_to_sequences(texts_for_tokenizer)  # 将文本转换为序列
print(sequences)

# 填充序列，使其长度一致
max_length = max(len(seq) for seq in sequences)  # 找到最长的序列长度
# padded_sequences = np.array([seq + [0] * (max_length - len(seq)) for seq in sequences])  # 使用0进行填充
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')  # 使用'post'进行填充
print(padded_sequences)

# 词汇表大小
vocab_size = len(word_index) + 1  # 加1是因为索引从1开始

# 定义模型
embedding_dim = 50  # 嵌入向量维度
model = Sequential()  # 定义一个顺序模型
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))  # 添加嵌入层
# model.add(Flatten())  # 添加展平层，将嵌入向量展平为一维，以便后续的全连接层处理。
# model.add(Dense(1, activation='softmax'))  # 添加全连接层，输出一个概率值

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 二分类问题

# 目标数据示例随机生成
y = np.random.randint(0, 2, size=(len(texts)))  # 生成与文本数量相同的随机二分类标签

# 训练模型
# model.fit(padded_sequences, y, epochs=100)  # 训练模型

# 第一个文本句子的嵌入向量
# print(padded_sequences)
input_tensor = tf.constant([padded_sequences[0]], dtype='int32')  # 创建一个输入张量
embedded_sequences = model.predict(input_tensor)  # 通过模型获取嵌入向量
print(embedded_sequences.shape)  # 输出：(batch_size, max_length, embedding_dim)
# 打印嵌入向量
print(embedded_sequences[0, 1, :])  # 打印'是'嵌入向量

# 第二个文本句子的嵌入向量
input_tensor = tf.constant([padded_sequences[1]], dtype='int32')  # 创建一个输入张量
embedded_sequences = model.predict(input_tensor)  # 通过模型获取嵌入向量
print(embedded_sequences.shape)  # 输出：(batch_size, max_length, embedding_dim)
# 打印嵌入向量
print(embedded_sequences[0, 1, :])  # 打印句子中'是'的嵌入向量，观察其与第一个文本句子的'是'嵌入向量是否相同

'''
# 定义一个向量
father = np.array([0.89, 0.01, 0, 0.002, 0.96])
mother = np.array([0.9, 0, 0, 0, 0.95])
banana = np.array([0, 0, 0.99, 0, 0.005])

# 计算向量的点积
dot_product = np.dot(father, mother)
print("向量的点积：", dot_product)

# 计算向量的模(长度)
father_norm = np.linalg.norm(father)
mother_norm = np.linalg.norm(mother)
print("向量的模：", father_norm, mother_norm)

# 计算余弦相似度
cosine_similarity = np.dot(father, mother) / (np.linalg.norm(father) * np.linalg.norm(mother))
print("余弦相似度：", cosine_similarity)

# 词向量的余弦相似度
with open('D:\RAG与LLM原理及实践\dataset\glove.6B\glove.6B.50d.txt', 'r', encoding='utf8') as f:
    words = set()  # 用来存储所有的单词
    word_to_vec_map = {}  # 用来存储单词到词向量的映射
    index = 1
    for line in f:
        line = line.strip().split()  # 按空格分割每一行
        if index == 1:
            print(line)  # 打印第一行
            print(len(line))  # 打印第一行的长度
            index += 1
        curr_word = line[0]  # 获取当前单词
        words.add(curr_word)  # 将当前单词添加到单词集合中
        word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)  # 将当前单词和对应的词向量添加到映射中

    i = 1
    words_to_index = {}  # 用来存储单词到索引的映射
    index_to_words = {}  # 用来存储索引到单词的映射
    for w in sorted(words):
        words_to_index[w] = i  # 将单词和索引添加到映射中
        index_to_words[i] = w  # 将索引和单词添加到映射中
        i += 1
return words_to_index, word_to_vec_map, index_to_words


def test(self, word1, word2):
    embedding1 = self.word_to_vec_map[word1]
    embedding2 = self.word_to_vec_map[word2]
    # 计算点积
    dot_product = np.dot(embedding1, embedding2)
    # 计算模
    embedding1_norm = np.linalg.norm(embedding1)
    embedding2_norm = np.linalg.norm(embedding2)
    # 计算余弦相似度
    cosine_similarity = dot_product / (embedding1_norm * embedding2_norm)
    return cosine_similarity

print(test(word1='banana', word2='mango'))
'''