import os, re, collections, time
from nltk.stem import PorterStemmer

path = "cranfield/"


def is_symbol(word):
    if word in [',', '.']:
        return True
    return False


def is_tag(string):
    match = re.search(r'<.*>', string)
    if match:
        return True
    return False


def remove_special_characters(word):
    word = word.replace(',', '').replace('/', '').replace('.', '').replace('-', '').replace('\'', '').replace('(', '')\
        .replace(')', '').replace('+', '').replace('*', '')
    return word


def get_only_once_count(dictionary):
    only_once_count = 0
    for k, v in dictionary.items():
        if v == 1:
            only_once_count += 1
    return only_once_count


def get_top30(dictionary):
    i = 0
    top_30 = {}
    for key, value in sorted(dictionary.items(), key=lambda item: item[1], reverse=True):
        if i <= 30:
            i += 1
            top_30[key] = value
    return top_30


list_of_tokens = []
word_per_doc = {}

start = time.time()
# Reading the files and creating raw tokens
for root, dir, files in os.walk(path):
    for file in files:
        raw_tokens = []
        with open(path + "/" + file) as f1:
            content = f1.read().split()
            for i in content:
                if not is_tag(i) and not is_symbol(i):
                    i = remove_special_characters(i)
                    if i is not '':
                        raw_tokens.append(i.lower())
            word_per_doc[file] = len(raw_tokens)
            list_of_tokens = list_of_tokens + raw_tokens

dict_of_tokens = collections.Counter(list_of_tokens)
print("TOKENIZATION Q1 : The number of tokens", len(list_of_tokens))
print("TOKENIZATION Q2 : The number of unique words :: ", len(dict_of_tokens.keys()))
print("TOKENIZATION Q3 : The number of words that occur only once :: ", get_only_once_count(dict_of_tokens))
print("TOKENIZATION Q4 : Top 30 most frequent words are :: ", get_top30(dict_of_tokens))
print("TOKENIZATION Q5 : The average number of tokens per document :: ", sum(word_per_doc.values()) / len(word_per_doc))
end = time.time()
print("time taken for tokenization", end-start)
print("=========================================")

# porter stemmer implementation
ps = PorterStemmer()
stemmed_tokens = []

for token in list_of_tokens:
    stemmed_tokens.append(ps.stem(token))

dict_of_stems = collections.Counter(stemmed_tokens)

print("STEMMING Q1 : Number of distinct stems :: ", len(dict_of_stems.keys()))
print("STEMMING Q2 : Number of stems that occur only once :: ", get_only_once_count(dict_of_stems))
print("STEMMING Q3 : Top 30 most frequent stems are :: ", get_top30(dict_of_stems))
print("STEMMING Q4 : The average stems per document :: ", sum(word_per_doc.values()) / len(word_per_doc))
