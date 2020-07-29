#!/usr/bin/python

import pickle
from gib import gib_detect_train

model_data = pickle.load(open('./model/gib_model.pki', 'rb'))

while True:
    # l = input('>>>')
    # l = 'i hope so True'
    l = 'ertrjiloifdfyyoiu'
    model_mat = model_data['mat']
    threshold = model_data['thresh']
    print(gib_detect_train.avg_transition_prob(l, model_mat) > threshold)
