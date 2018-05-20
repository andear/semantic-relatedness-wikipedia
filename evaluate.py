import sys
from functools import cmp_to_key
import numpy as np
import numpy.linalg as LA

def get_rank(score):
    '''Given a list and give out the rank result of the list'''
    tmp = [(i, item) for i, item in enumerate(score)]
    tmp = sorted(tmp, key=cmp_to_key(lambda x, y: (x[1] > y[1]) - (x[1] < y[1])))
    print(type(tmp))
    print(len(tmp))

    res = [0 for i in range(len(tmp))]
    for i, (pos, score) in enumerate(tmp):
        res[pos] = i
    return res


def load_vector(filename):
    '''Load the word vector of the filenmae'''
    data = {}
    with open(filename, "r") as f:
        for l in f:
            words = l.strip().split()
            word = words[0]
            res = np.array([float(item) for item in words[1:]])
            data[word] = res
    return data


def load_wordsim353(filename):
    ''' load the wordsim 353 test
    the test format:
    worda wordb score'''
    words = []
    with open(filename) as f:
        for l in f:
            words.append(l.strip().split())
    test_words = [(item[0], item[1]) for item in words]
    answer = [float(item[2]) for item in words]
    return test_words, answer


def similarity(word1, word2, word2vector):
    '''get the cosine similarity of the word1 and word2'''
    if word1 in word2vector:
        vector1 = word2vector[word1]
    else:
        print("key do not exist:" + word1)
        return 0

    if word2 in word2vector:
        vector2 = word2vector[word2]
    else:
        print("key do not exist:" + word2)
        return 0

    res = np.dot(vector1, vector2) / (LA.norm(vector1) * LA.norm(vector2))
    res = float(res)
    return res


def get_score(test_words, word2vector):
    '''test the word2vector using test_words'''
    return [similarity(word1, word2, word2vector) for word1, word2 in test_words]

def get_corr(listA, listB):
    '''get teh correlation of the listA and List B'''
    x_bar = 0.0
    y_bar = 0.0
    for a, b in zip(listA, listB):
        x_bar += a
        y_bar += b
    x_bar *= (1. / float(len(listA)))
    y_bar *= (1. / float(len(listB)))
    res1 = 0.0
    res2 = 0.0
    res3 = 0.0
    for x, y in zip(listA, listB):
        res1 += ((x - x_bar) * (y - y_bar))
        res2 += ((x - x_bar) * (x - x_bar))
        res3 += ((y - y_bar) * (y - y_bar))
    return res1/np.sqrt(res2 * res3)


def test_wordsim353(vector_filename, test_filename):
    ''' test the result of the vector_filename and the test_filename'''
    word2vector = load_vector(vector_filename)
    test_words, answer = load_wordsim353(test_filename)
    my_answer = get_score(test_words, word2vector)

    answer_rank = get_rank(answer)
    my_rank = get_rank(my_answer)
    return get_corr(my_rank, answer_rank)

if __name__ == "__main__":
    vector_filename1 = "./glove.6B/glove.6B.50d.txt"
    vector_filename2 = "./glove.6B/glove.6B.100d.txt"
    vector_filename3 = "./glove.6B/glove.6B.200d.txt"
    vector_filename4 = "./glove.6B/glove.6B.300d.txt"

    test_filename = "./wordsim353.txt"

    print(test_wordsim353(vector_filename1, test_filename))
    print("file1 done...")
    print(test_wordsim353(vector_filename2, test_filename))
    print("file2 done...")
    print(test_wordsim353(vector_filename3, test_filename))
    print("file3 done...")
    print(test_wordsim353(vector_filename4, test_filename))
    print("file4 done...")