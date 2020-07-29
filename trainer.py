import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
import pickle
import random

random.seed(12345)

raw = pd.read_csv('./tmp/vectorized_feature_w_ranks_norm.txt')

X = raw.loc[:, 'bi_rank':'vowel_ratio'].as_matrix()
Y = raw.loc[:, 'class'].as_matrix()

domains = raw.ix[:, 'ip'].as_matrix()

from sklearn import linear_model, decomposition, datasets

n_samples, n_features = X.shape
p = list(range(n_samples))  # Shuffle samples


# random initialization


classifier = SVC(kernel='linear', probability=True, random_state=0)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

precision_list = []
recall_list = []
area_list = []
fpr_list = []
tpr_list = []
roc_auc_list = []
accuracy_list = []

from sklearn.metrics import roc_curve, auc

max_ac = 0
for i in range(10):  # 10 fold cross-validation
    print('x-validation round %d' % i)
    random.seed(i)
    p = random.sample(p, int(n_samples / 10))
    # random.shuffle(p)
    XX, yy = X[p], Y[p]
    cut_off = int(len(XX) / 5) * 4
    probas_ = classifier.fit(XX[:cut_off], yy[:cut_off]).predict(XX[cut_off:])
    # precision, recall, thresholds = precision_recall_curve(yy[cut_off:], probas_)
    # fpr, tpr, thresholds = roc_curve(yy[cut_off:], probas_)
    # roc_auc = auc(fpr, tpr)
    # area = auc(recall, precision)
    # precision_list.append(precision)
    # recall_list.append(recall)
    # area_list.append(area)
    # fpr_list.append(fpr)
    # tpr_list.append(tpr)
    # roc_auc_list.append(roc_auc)
    # pred = [int(i > 0.5) for i in probas_]
    # accuracy_list.append(accuracy_score(yy[cut_off:], pred, normalize=True))
    ac_score = metrics.accuracy_score(yy[cut_off:], probas_)
    print(ac_score)
    if ac_score > max_ac:
        with open('./model/svm_738.pickle', 'wb') as fw:
            pickle.dump(classifier, fw)
            print("第%d次训练，保存模型" % i)
        max_ac = ac_score
