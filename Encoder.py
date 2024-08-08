import jieba
import numpy as np
from keras.preprocessing.text import Tokenizer

texts = ['张某是一位智能编程助手', '这是测试文本']
tokenized_texts = [list(jieba.cut(text)) for text in texts]
text_for_tokenizer = [' '.join(tokenized_text) for tokenized_text in tokenized_texts]

# 分词和索引化
tokenizer = Tokenizer(num_words=1000)

