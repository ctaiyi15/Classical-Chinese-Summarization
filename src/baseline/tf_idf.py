from collections import Counter
import math

def get_tfidf(text, sen_num):
    tf_idf = []
    sen_word_bag_list = []
    tf_list = []
    for sen in text:
        sen = sen.lower().split()
        sen_word_count = Counter(sen)
        sen_word_bag = set(sen)
        sen_word_bag_list.append(Counter(sen_word_bag))
        sen_tf = [(word,sen_word_count[word]/len(sen)) for word in sen_word_count]
        tf_list.append(sen_tf)
    word_count = sum(sen_word_bag_list, Counter())
    idf = {word:math.log(sen_num/count) for word,count in word_count.items()}
    tf_idf = [[(word[0],word[1]*idf[word[0]]) for word in tf] for tf in tf_list]
    return tf_idf, word_count
