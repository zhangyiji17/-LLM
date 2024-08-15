import jieba
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
import tensorflow as tf

texts = ['张某是一位智能编程助手', '这里是测试文本']
tokenized_texts = [list(jieba.cut(text)) for text in texts]
text_for_tokenizer = [' '.join(tokenized_text) for tokenized_text in tokenized_texts]

# 分词和索引化
tokenizer = Tokenizer(num_words=1000)  # 设置词汇表大小为1000
tokenizer.fit_on_texts(text_for_tokenizer)  # 对文本进行分词和索引化
word_index = tokenizer.word_index  # 获取词汇表索引
sequences = tokenizer.texts_to_sequences(text_for_tokenizer)  # 将文本转换为序列
print(sequences)

# 填充序列
max_length = max(len(sequence) for sequence in sequences)
print(max_length)
padded_sequences = np.array([sequence + [0] * (max_length - len(sequence)) for sequence in sequences])  # 填充序列


# 词汇表大小
vocab_size = len(word_index) + 1  # 加1是因为索引从1开始

# 定义模型
embedding_dim = 50  # 嵌入维度
model = Sequential()  # 顺序模型
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))  # 嵌入层
# model.add(Flatten())  # 展平层，将嵌入向量展平，用于后续的全连接层
# model.add(Dense(1, activation='softmax'))  # 全连接层，输出维度为1，使用softmax激活函数，用于分类

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 使用adam优化器和交叉熵损失函数，评估指标为准确率

# 目标数据
targets = np.random.randint(0, 2, size=(len(texts), ))  # 随机生成目标数据，这里假设有2个类别

# 训练模型
# model.fit(padded_sequences, targets, epochs=10)  # 训练模型，设置训练轮数为10，批量大小为32

# 词向量
input_tensor = tf.constant([padded_sequences[1]], dtype='int32')  # 创建一个输入张量
embedded_sequences = model.predict(input_tensor)  # 通过模型获取嵌入向量
print(embedded_sequences.shape)  # 输出：(batch_size, max_length, embedding_units)

# 打印嵌入向量
for i in range(max_length):
    print(embedded_sequences[0, i, :])

gru = tf.keras.layers.GRU(units=50,  # 定义GRU层，128个单元
                          return_sequences=True,  # 返回序列
                          return_state=True,  # 返回状态
                          recurrent_initializer='glorot_uniform',  # 使用均匀分布初始化权重
                          )

embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)  # 定义嵌入层，输入维度为词汇表大小


x = embedding(padded_sequences)  # 将输入序列转换为嵌入向量
print(x.shape)

position = tf.range(start=0, limit=max_length, dtype=tf.float32)[:, tf.newaxis]  # 创建位置编码

# 使用正弦和余弦函数计算位置编码
div_term = tf.math.pow(10000.0, tf.range(0, embedding_dim, 2, dtype=tf.float32) / embedding_dim)  # 计算除数
pos_encoding = tf.concat([tf.sin(position * div_term), tf.cos(position * div_term)], axis=-1)  # 计算位置编码

print(pos_encoding)  # 输出：(max_length, embedding_dim)

'''
参数解释：
    vocab_size：词汇表大小，即词汇表中唯一词的数量。
    embedding_dim：嵌入维度，即每个词将被表示为的向量维度。embedding_units
    encoding_dim：编码器LSTM层的输出维度。encoding_units
    batch_size：训练时每个批次的数据量。
'''

'''
LSTM(长短期记忆)：
    LSTM是一种特殊的RNN（循环神经网络），它通过引入门控机制来缓解梯度消失和梯度爆炸的问题，从而更好地处理长序列数据。
    LSTM通过三个门（输入门、遗忘门和输出门）来控制信息的流动，从而能够更好地捕捉长序列中的时间依赖关系。
四个部分：输入门、遗忘门、输出门和细胞状态。
    输入门：决定哪些新信息存储在细胞状态中。
    遗忘门：决定丢弃细胞状态中的部分信息。
    细胞状态：存储长期信息，类型于‘记忆‘。
    输出门：决定将哪些信息输出到隐藏状态。
'''

'''
GRU（门控循环单元）：
    GRU是LSTM的一种变体，它通过合并输入门和遗忘门，以及引入一个新的门（更新门）来简化LSTM的结构。
两个部分：更新门和重置门。
    更新门：决定如何更新隐藏状态，同时控制哪些信息需要被遗忘以及哪些新信息需要被添加。
    重置门：决定如何重置隐藏状态，丢弃哪些与上一个时间步相关的信息。
    GRU没有单独的细胞状态，而是将信息存储在隐藏状态中。
'''
