import pandas as pd
import numpy as np

vec_feature = pd.read_csv('./data/conficker_alexa_training.txt', sep='\t', header=None)
index = np.array(range(vec_feature.shape[0]))

np.random.seed(123456)
np.random.shuffle(index)

train_data = vec_feature.iloc[index[:-10000], :]
test_data = vec_feature.iloc[index[-10000:], :]

train_data.to_csv('./tmp/train_data.txt', index=False, header=False, sep='\t')
test_data.to_csv('./tmp/test_data.txt', index=False, header=False, sep='\t')