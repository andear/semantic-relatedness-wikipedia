#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xiaohan, Pan Yang (panyangnlp@gmail.com)

import gensim
from gensim.models.word2vec import LineSentence
import logging
import multiprocessing
import os
import re
import sys
import string 
from pattern.en import tokenize
from time import time
import io

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

def cleanhtml(raw_html):
    cleanr = re.compile('<a href="')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = re.sub(r'">.*?>', '', cleantext)
    cleantext = re.sub(r'%\d{2}', '', cleantext)
    # cleantext = re.sub(r'--', '', cleantext)
    # cleantext = re.sub(r'- ', '', cleantext)
    # print cleantext
    return cleantext

 
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for filename in files:
                file_path = root + '/' + filename
                for line in open(file_path):
                    sline = line.strip()
                    if sline == "":
                        continue
                    rline = cleanhtml(sline)
                    tokenized_line = ' '.join(tokenize(rline))
                    is_alpha_word_line = [word for word in
                                          tokenized_line.lower().split()
                                          if word.isalpha()]
                    yield is_alpha_word_line
 
 
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Please use python train_with_gensim.py data_path"
        exit()
    data_path = sys.argv[1]
    begin = time()
 
    sentences = MySentences(data_path)
    #sg = 1 -> using, sample>0 -> downsampling
    model = gensim.models.Word2Vec(sentences,
                                   size=200,
                                   window=10,
                                   min_count=5,
                                   sg=1,
                                   sample = 1e-5,
                                   workers=multiprocessing.cpu_count())
    model.save("data/model/word2vec_gensim")
    model.wv.save_word2vec_format("data/model/word2vec_org",
                                  "data/model/vocabulary",
                                  binary=False)
 
    end = time()
    print "Total procesing time: %d seconds" % (end - begin)