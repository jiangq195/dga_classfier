import random

import math
import tldextract

from feat_extractor import bigrams, trigrams, extract_feature
from collections import Counter
import numpy as np
import pandas as pd
import pickle
import random
from itertools import groupby

from feat_normalizer import feat_norm
from feat_vectorizer import get_vec

feature_header = ['tld', 'entropy', 'len', 'norm_entropy', 'vowel_ratio', 'digit_ratio', 'repeat_letter',
                  'consec_digit', 'consec_consonant', 'gib_value', 'hmm_log', 'uni_rank', 'bi_rank', 'tri_rank',
                  'uni_std', 'bi_std', 'tri_std']


def count_vowels(word):  # how many a,e,i,o,u
    vowels = list('aeiou')
    return sum(vowels.count(i) for i in word.lower())


def count_digits(word):  # how many digits
    digits = list('0123456789')
    return sum(digits.count(i) for i in word.lower())


def count_repeat_letter(word):  # how many repeated letter
    count = Counter(i for i in word.lower() if i.isalpha()).most_common()
    cnt = 0
    for letter, ct in count:
        if ct > 1:
            cnt += 1
    return cnt


def consecutive_digits(word):  # how many consecutive digit
    cnt = 0
    digit_map = [int(i.isdigit()) for i in word]
    consecutive = [(k, len(list(g))) for k, g in groupby(digit_map)]
    count_consecutive = sum(j for i, j in consecutive if j > 1 and i == 1)
    return count_consecutive


def consecutive_consonant(word):  # how many consecutive consonant
    cnt = 0
    # consonant = set(['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'z'])
    consonant = set(
        ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'])
    digit_map = [int(i in consonant) for i in word]
    consecutive = [(k, len(list(g))) for k, g in groupby(digit_map)]
    count_consecutive = sum(j for i, j in consecutive if j > 1 and i == 1)
    return count_consecutive


def hmm_prob(domain):
    bigram = [''.join((i, j)) for i, j in bigrams(domain) if not i == None]
    prob = transitions[''][bigram[0]]
    for x in range(len(bigram) - 1):
        next_step = transitions[bigram[x]][bigram[x + 1]]
        prob *= next_step

    return prob

def extract_tld(domain):
    """提取顶级域名"""
    with open('./data/tld_list.txt', 'r', encoding='utf8') as f:
        tlds = list('.' + t.strip().strip('.') + '.' for t in f)  # for domain match, add dot as prefix and postfix
    match = [i for i in tlds if i in domain]
    if len(match) > 0:
        if len(match) > 1:
            pass
        for i in sorted(match, key=lambda x: len(x), reverse=True):
            if i == domain[-(len(i)):]:  # 最大化匹配到末尾
                return (domain, i)
    else:
        return (domain, 'NONE')


def decimal_process(feature):
    """格式化为训练输入数据格式"""

    feature[1] = feature[1].lower()
    feature[2] = round(feature[2], 3)
    feature[3] = round(feature[3], 1)
    feature[4] = round(feature[4], 3)
    for i in range(5, len(feature) - 1):
        feature[i] = round(feature[i], 2)

    return feature[1:-1]


def normalizer(feature):
    train_norm_args, _ = feat_norm()
    feat_matrix = pd.DataFrame()
    feat_matrix[0] = feature.iloc[:, 0]
    for i in range(1, feature.shape[1]):
        cloumn = feature.iloc[:, i]
        mean_ = train_norm_args[i - 1][0]
        max_ = train_norm_args[i - 1][1]
        min_ = train_norm_args[i - 1][2]
        feat_matrix[i] = round((cloumn - mean_) / (max_ - min_), 2)

    return feat_matrix


def extract_feature(domain):
    strip_domain = domain
    # if not tld=='NONE':
    #    strip_domain = domain[:-len(tld)]
    # strip_domain=strip_domain.strip('.')
    # main_domain = '$'+strip_domain.split('.')[-1]+'$'
    ext = tldextract.extract(strip_domain)  # user tld extractor for more precision
    if len(ext.domain) > 4 and ext.domain[:4] == 'xn--':  # remove non-ascii domain
        return None
    main_domain = '$' + ext.domain + '$'  # add begin and end
    hmm_main_domain = '^' + domain.strip('.') + '$'  # ^ and $ of full domain name for HMM
    tld = ext.suffix
    has_private_tld = 0
    # check if it is a private tld

    private_tld_file = open('./data/private_tld.txt', 'r')
    private_tld = set(f.strip() for f in private_tld_file)  # black list for private tld
    private_tld_file.close()

    if tld in private_tld:
        has_private_tld = 1
        tld_list = tld.split('.')  # quick hack: if private tld, use its last part of top TLD
        tld = tld_list[-1]
        main_domain = '$' + tld_list[-2] + '$'  # and overwrite the main domain
    bigram = [''.join(i) for i in bigrams(main_domain)]  # extract the bigram
    trigram = [''.join(i) for i in trigrams(main_domain)]  # extract the bigram
    f_len = float(len(main_domain))
    count = Counter(i for i in main_domain).most_common()  # unigram frequency
    entropy = -sum(j / f_len * (math.log(j / f_len)) for i, j in count)  # shannon entropy

    n_gram_file = open('./tmp/n_gram_rank_freq.txt', 'r')
    gram_rank_dict = dict()
    for i in n_gram_file:
        cat, gram, freq, rank = i.strip().split(',')
        gram_rank_dict[gram] = int(rank)
    n_gram_file.close()

    unigram_rank = np.array([gram_rank_dict[i] if i in gram_rank_dict else 0 for i in main_domain[1:-1]])
    bigram_rank = np.array([gram_rank_dict[''.join(i)] if ''.join(i) in gram_rank_dict else 0 for i in
                            bigrams(main_domain)])  # extract the bigram
    trigram_rank = np.array([gram_rank_dict[''.join(i)] if ''.join(i) in gram_rank_dict else 0 for i in
                             trigrams(main_domain)])  # extract the bigram

    # linguistic feature: % of vowels, % of digits, % of repeated letter, % consecutive digits and % non-'aeiou'
    vowel_ratio = count_vowels(main_domain) / f_len
    digit_ratio = count_digits(main_domain) / f_len
    repeat_letter = count_repeat_letter(main_domain) / f_len
    consec_digit = consecutive_digits(main_domain) / f_len
    consec_consonant = consecutive_consonant(main_domain) / f_len

    # probability of staying in the markov transition matrix (trained by Alexa)
    hmm_prob_ = hmm_prob(hmm_main_domain)
    if hmm_prob_ < math.e ** hmm_prob_threshold:  # probability is too low to be non-DGA
        hmm_log_prob = -999.
    else:
        hmm_log_prob = math.log(hmm_prob_)

    # advanced linguistic feature: pronouncable domain
    gib_value = int(gib_detect_train.avg_transition_prob(main_domain.strip('$'), model_mat) > threshold)

    return [domain, tld, entropy, f_len, entropy / f_len, vowel_ratio,
            digit_ratio, repeat_letter, consec_digit, consec_consonant, gib_value, hmm_log_prob,
            ave(unigram_rank), ave(bigram_rank), ave(trigram_rank),
            std(unigram_rank), std(bigram_rank), std(trigram_rank),
            has_private_tld]


def main(domains):
    temp = []
    ascii_domain_idx = np.array(range(len(domains)), dtype=int)
    no_ascii_idx = []
    for idx in range(len(domains)):
        temp1 = extract_feature(domains[idx])
        if not temp1:
            # 跳过非ascii domain
            no_ascii_idx.append(idx)
            print('%s non-ascii domain' % domains[idx])
            continue
        temp2 = decimal_process(temp1)
        temp.append(temp2)

    test_features_dec = pd.DataFrame(temp)
    ascii_domain_idx = np.delete(ascii_domain_idx, no_ascii_idx)
    test_features = [dict(zip(feature_header, item)) for item in normalizer(test_features_dec).values.tolist()]
    _, vec, _ = get_vec()

    test_features = vec.transform(test_features).todense()

    with open('./model/svm_738.pickle', 'rb') as f:
        model = pickle.load(f)

    predict = model.predict(test_features)

    return {
        'predict': predict,
        'ascii_domain_idx': ascii_domain_idx,
        'no_ascii_idx': no_ascii_idx
    }


if __name__ == '__main__':
    test_samples = pd.read_csv('./tmp/test_data.txt', sep='\t', header=None)
    test_X = test_samples[0].values
    test_y = test_samples[1].values
    result = main(test_X)
    test_y = test_y[result['ascii_domain_idx']]
    res = (result['predict'] == test_y).sum()
    print(res / 10000)
