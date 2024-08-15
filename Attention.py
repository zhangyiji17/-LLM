import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, embedding_units, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads    # 注意力头的数量
        self.embedding_units = embedding_units  # 嵌入维度

        # 分割units为num_heads个部分
        assert embedding_units % num_heads == 0  # 确保embedding_units可以被num_heads整除
        self.depth = embedding_units // num_heads

        # 初始化权重矩阵
        self.Wq = tf.keras.layers.Dense(embedding_units)  # query的权重矩阵
        self.Wk = tf.keras.layers.Dense(embedding_units)  # key的权重矩阵
        self.Wv = tf.keras.layers.Dense(embedding_units)  # value的权重矩阵

        self.dense = tf.keras.layers.Dense(embedding_units)  # 输出层的权重矩阵

    def split_heads(self, x, batch_size):
        # 将输入张量分割成多个注意力头
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # 分割输入张量的最后一个维度到(num_heads, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # 转置为(batch_size, num_heads, seq_len, depth)

    def call(self, inputs, training=False):
        # inputs: [batch_size, seq_len, embedding_units]
        batch_size = tf.shape(inputs)[0]  # 获取输入张量的batch_size
        sequence_length = tf.shape(inputs)[1]  # 获取输入张量的sequence_length

        # 计算query, key, value
        q = self.split_heads(self.Wq(inputs), batch_size)  # [batch_size, num_heads, seq_len, depth]
        k = self.split_heads(self.Wk(inputs), batch_size)  # [batch_size, num_heads, seq_len, depth]
        v = self.split_heads(self.Wv(inputs), batch_size)  # [batch_size, num_heads, seq_len, depth]

        # 计算注意力分数  [batch_size, num_heads, seq_len, seq_len]
        scaled_attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        # 应用softmax函数计算注意力权重  # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)
        print(attention_weights)

        # 上下文向量 [batch_size, num_heads, seq_len, depth]
        output = tf.matmul(attention_weights, v)

        # 将多个注意力头合并为一个张量  [batch_size, seq_len, num_heads, depth]
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        # 合并深度维度  [batch_size, seq_len, embedding_units]
        output = tf.reshape(output, (batch_size, sequence_length, self.embedding_units))

        # 通过全连接层得到最终的输出  [batch_size, seq_len, embedding_units]
        output = self.dense(output)

        return output


class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_units, num_heads):
        super(SimpleAttention, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=1000, output_dim=embedding_units)
        self.self_attention = Attention(embedding_units=embedding_units, num_heads=num_heads)
        self.output_layer = tf.keras.layers.Dense(1)  # 全连接层
        # self.dense = tf.keras.layers.Dense(embedding_units)  # 全连接层

    def call(self, inputs, training=False):
        # 假设inputs: [batch_size, seq_len]
        x = self.embedding_layer(inputs)  # [batch_size, seq_len, embedding_units]
        # 应用自注意力机制
        x = self.self_attention(x, training=training)  # [batch_size, seq_len, embedding_units]
        # 其它层如位置编码，前馈网络等，这里不做添加
        # 直接对序列最后一个元素进行平均池化简化操作示例
        pooled = tf.reduce_mean(x, axis=1)  # [batch_size, embedding_units]
        output = self.output_layer(pooled)  # [batch_size, 1]
        return output


# 测试
if __name__ == '__main__':
    # 嵌入维度
    embedding_units = 128
    # 多头注意力机制
    num_heads = 8
    # 输入序列长度
    sequence_length = 10
    # 批量大小
    batch_size = 2
    # 构造输入数据，这里使用随机生成的数据
    inputs = tf.random.uniform((batch_size, sequence_length), minval=0, maxval=1000, dtype=tf.int32)
    #print(inputs)
    # 实例化模型
    model = SimpleAttention(embedding_units, num_heads)
    # 调用模型进行前向传播
    output = model(inputs, training=True)
    # 打印输出结果
    # print(output.shape)  # [batch_size, 1]
    print(output)


