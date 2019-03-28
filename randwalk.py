# -*- coding: utf-8 -*-
import numpy
import re
import math
import jieba

def get_tfisf_data(raw_article_sents):
    sentenceTFISFVectors = []
    sentencesLen = []

    # calculate tfisf and the sentence vector
    word_list = []
    total_sens = len(raw_article_sents)
    isf = {}

    for sen in raw_article_sents:
        # get word list and setencelen
        # raw article sents 是已经拉平的文章句子，利用doc_indices就可以对应到文章原结构
        # 输入文件已经分好词了，直接用空格分割就行
        list_of_words = sen.split(" ")
        list_of_words = [x for x in list_of_words if x != '']
        sentencesLen.append(len(list_of_words))

        uniq_words = set(list_of_words)
        for w in uniq_words:
            isf[w] = isf.get(w, 0.0) + 1.0

        for w in uniq_words:
            if w not in word_list:
                word_list.append(w)

    for k in isf:
        isf[k] = 1 + math.log(total_sens / isf[k])

    for sen in raw_article_sents:
        # get tf & sentence vec
        tf = {}
        list_of_words = sen.split(" ")
        list_of_words = [x for x in list_of_words if x != '']

        for w in list_of_words:
            tf[w] = tf.get(w, 0.0) + 1.0

        length = float(len(list_of_words))

        for k in tf:
            tf[k] = tf[k] / length

        vec = [0.0] * len(word_list)
        for k in list_of_words:
            i = word_list.index(k)
            vec[i] = isf[k]
            vec[i] = vec[i]*tf[k]

        sentenceTFISFVectors.append(vec)

    return sentenceTFISFVectors, sentencesLen

def rw_calculator(sentenceTFISFVectors, indoc, sentencesLen, mu, batch):
    N = len(sentenceTFISFVectors)
    # sentenceNorm: the length of sentence vectors
    sentenceNorm = numpy.zeros((N, 1))
    for i in range(0, N):
        sentenceNorm[i, 0] = numpy.linalg.norm(numpy.mat(sentenceTFISFVectors[i]))

    # W: the sentence affinity graph
    # Wintra: the affinity graph of sentences oriented to the same document
    # Winter: the affinity graph of sentences oriented to the different document
    # Wintra and Winter offer some flexibility
    Wintra = numpy.zeros((N, N))
    Winter = numpy.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            if i<j:
                v1=numpy.mat(sentenceTFISFVectors[i])
                v2=numpy.mat(sentenceTFISFVectors[j])
                sim = v1*v2.T/sentenceNorm[i]/sentenceNorm[j]
                # indoc: the indicator whether two sentences are in the same document
                if indoc[i] == indoc[j]:
                    if sim > 0.05:
                        Wintra[i, j] = sim
                    else:
                        Wintra[i, j] = 1.0 / N
                    Winter[i, j] = 0.0
                else:
                    Wintra[i, j] = 0.0
                    if sim > 0.05:
                        Winter[i, j] = sim
                    else:
                        Winter[i, j] = 1.0 / N
            else:
                Wintra[i, j] = Wintra[j, i]
                Winter[i, j] = Winter[j, i]
    W = Wintra + Winter    

    # D: the diagonal matrix of W
    # rowsum: the sum of the elements in each row
    # maxrowsum: the maximal element sum of the row
    D = numpy.zeros((N, N))
    D = numpy.mat(D)
    rowsum = []
    maxrowsum = 0
    for i in range(0, N):
        rs = 0
        for j in range(0, N):
            rs = rs + W[i, j]
        rowsum.append(rs)
        if maxrowsum < rs:
            maxrowsum = rs
    for i in range(0, N):
        D[i,i] = maxrowsum

    # y: the prior vector (proportional to the position of the sentence) (sum(y)=1)
    l = list()
    l.append(1)
    ini = 0
    for i in range(1, N):
        if indoc[i] == indoc[i-1]:
            l.append(2 ** ( - ( i - ini )))
        else:
            l.append(1)
            ini = i
    y = numpy.mat(l).T
    y = y / sum(y)

    # e: the vector with all one elements
    e = numpy.ones((N, 1))

    # A1: probability transition matrix of affinity-preserving random walk
    # mu: the damping factor
    min_error = math.exp(-4)
    A1 = numpy.dot(D.I, W).T
    A1 = mu * A1 + (1 - mu) * numpy.dot(y, e.T)
    # v: the vector for iteration of affinity-preserving random walk (sum(v)=1)
    v = numpy.ones((N, 1))
    v = v / sum(v)
    # sentencesLen: the length of each sentence
    for c in range(0, batch):
        if c > 30:
            # virtualSummary: the summary based on transient distribution
            virtualSummary = summarize(v, A1, sentencesLen)
            # D: the adjusted diagonal matrix (local normalization for the summary sentences)
            D = adjust(virtualSummary, rowsum, maxrowsum, D)
            # Adjust the probability transition matrix A1 according to D
            A1 = mu * (numpy.dot(D**(-1), W).T) + (1 - mu) * numpy.dot(y, e.T)
        # Iterate according to A1
        v_ = v
        v = A1 * v
        # Compute the conditional distribution
        v = v / sum(v)
        #if c < 10:
        if numpy.linalg.norm(v-v_) < min_error:
            # print("c: ", c)
            break

    score = (v.T).tolist()[0]
    return score
