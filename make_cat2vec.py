import numpy as np

fname = "embodiedscan_infos\embodiedscan_infos_val_full.pkl"
data = np.load(fname, allow_pickle=True)
categories = data['metainfo']['categories'].keys()
words_shown = set()
for category in categories:
    words_shown.update(category.split())
print(len(words_shown)) #292
del data
print(len(categories)) #288

# print(categories)

glove_file_name = "glove.42B.300d.txt"
# keep only words in categories
word_to_vec_map = {}
with open(glove_file_name, 'r', encoding="utf8") as f:
    for line in f:
        word, vec = line.split(' ', 1)
        if word in words_shown:
            word_to_vec_map[word] = list(map(float, vec.split()))
category2vec = {}
for category in categories:
    category_vec = np.zeros(300)
    for word in category.split():
        if word in word_to_vec_map:
            category_vec += np.array(word_to_vec_map[word])
    category_vec /= len(category.split())
    category2vec[category] = category_vec.tolist()
import json
with open('cat2vec.json', 'w') as f:
    json.dump(category2vec, f)