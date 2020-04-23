from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from prettytable import PrettyTable
import collections, operator, os, re, sys, time

base_path = "../../../Hw1/data/"


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


stop_words = []
with open(base_path+"stopwords") as file:
    stop_words = file.read().split()


def file_preprocess(content):
    raw_tokens = []
    stop_words_count = 0
    for i in content:
        if not is_tag(i) and not is_symbol(i):
            i = remove_special_characters(i)
            if i is not '':
                if i not in stop_words:
                    raw_tokens.append(i.lower())
                else:
                    stop_words_count += 1
    return raw_tokens, stop_words_count


ps = PorterStemmer()
def get_list_of_stems(list_of_tokens):
    stemmed_tokens = []
    for token in list_of_tokens:
        stemmed_tokens.append(ps.stem(token))

    return stemmed_tokens


lemmatizer = WordNetLemmatizer()

def get_list_of_lemmas(list_of_tokens):
    lemmas = []
    for word in list_of_tokens:
        lemmas.append(lemmatizer.lemmatize(word))
    return lemmas


def get_posting_list(posting_list, doc_id, tf, max_tf, doc_len):
    df = posting_list[0] + 1
    tuple = [(doc_id, tf, max_tf, doc_len)]
    listt = posting_list[1] + tuple
    value = (df, listt)
    return value


def create_posting_list(doc_id, tf, max_tf, doc_len):
    tuple = (doc_id, tf, max_tf, doc_len)
    posting_list = [tuple]
    value = (1, posting_list)
    return value

dict_of_stems = {}
dict_of_lemmas = {}

#########################################################
# Index Construction
#########################################################

# Reading the files and creating raw tokens
time_for_index1 = 0
time_for_index2 = 0
for root, dir, files in os.walk(base_path+"cranfield"):
    for file in files:
        with open(base_path+"cranfield" + "/" + file) as f1:
            content = f1.read().split()
            raw_tokens, stop_words_count = file_preprocess(content)

            raw_tokens_count = len(raw_tokens)
            doc_len = raw_tokens_count + stop_words_count

            start = time.time()
            file_tokens = collections.Counter(get_list_of_stems(raw_tokens))
            for token in file_tokens.keys():
                doc_ids = []
                if token not in dict_of_stems.keys():
                    posting_list = create_posting_list(file.replace('cranfield',''), file_tokens[token], max(file_tokens.items(), key=operator.itemgetter(1))[1], doc_len)
                else:
                    posting_list = get_posting_list(dict_of_stems[token], file.replace('cranfield',''), file_tokens[token], max(file_tokens.items(), key=operator.itemgetter(1))[1], doc_len)

                dict_of_stems[token] = posting_list
            time_for_index2 += time.time() - start

            start = time.time()
            file_tokens = collections.Counter(get_list_of_lemmas(raw_tokens))
            for token in file_tokens.keys():
                doc_ids = []
                if token not in dict_of_lemmas.keys():
                    posting_list = create_posting_list(file.replace('cranfield',''), file_tokens[token], max(file_tokens.items(), key=operator.itemgetter(1))[1], doc_len)
                else:
                    posting_list = get_posting_list(dict_of_lemmas[token], file.replace('cranfield',''), file_tokens[token], max(file_tokens.items(), key=operator.itemgetter(1))[1], doc_len)

                dict_of_lemmas[token] = posting_list
            time_for_index1 += time.time() - start

Index_Version1_uncompress = dict(sorted(dict_of_lemmas.items(), key=lambda x: x[0]))
Index_Version2_uncompress = dict(sorted(dict_of_stems.items(), key=lambda x: x[0]))

print("1. Time taken to build Index_Version1 is", time_for_index1)
print("1. Time taken to build Index_Version2 is", time_for_index2)

with open("../out/Index_Version1_uncompress.txt", 'w') as handler:
    handler.write(str(Index_Version1_uncompress))
handler.close()

with open("../out/Index_Version2_uncompress.txt", 'w') as handler:
    handler.write(str(Index_Version2_uncompress))
handler.close()

print("2. Size of Index_Version1_uncompress", os.path.getsize("../out/Index_Version1_uncompress.txt"))
print("3. Size of Index_Version2_uncompress", os.path.getsize("../out/Index_Version2_uncompress.txt"))

#########################################################
# Index Compression
#########################################################
# Posting Files Compression and dictionary compression
#########################################################


def get_unary(number):
    a = ['1' for i in range(number)] + ['0']
    unar = ""
    return unar.join(a)


def get_gamma_code(number):
    offset = str(bin(number)).split('b')[1][1:]
    unary = get_unary(len(offset))
    return (unary + offset)


def get_gamma_codes(gaps):
    gamma_code = []
    for number in gaps:
        gamma_code.append(get_gamma_code(number))
    return gamma_code


def get_delta_code(gaps):
    delta_codes = []
    for number in gaps:
        binary = str(bin(number)).split('b')[1]
        gamma = get_gamma_code(len(binary))
        offset = binary[1:]
        delta_codes.append(gamma + offset)
    return delta_codes


Index_Version1_compress = {}
block_compressed_terms = []
to_be_compressed = []
count = 0
for term in Index_Version1_uncompress.keys():
    # ## compress terms now
    if count < 4:
        to_be_compressed.append(term)
        count += 1
    elif count == 4:
        terms = "".join(to_be_compressed)
        terms2 = str(len(terms)) + terms
        block_compressed_terms.append(terms2)
        count = 0
        to_be_compressed = []

    posting_list = Index_Version1_uncompress[term][1]
    doc_ids = []
    for listt in posting_list:
        doc_ids.append(int(listt[0]))
    doc_ids.sort()
    gaps = []
    gaps.append(doc_ids[0])
    for i in range(1, len(doc_ids)):
        gaps.append(doc_ids[i] - doc_ids[i-1])
    Index_Version1_compress[term] = get_gamma_codes(gaps)

with open("../out/Index_Version1_compress", 'w') as handler:
    handler.write("".join(block_compressed_terms))
handler.close()

with open("../out/Index_Version1_compress", 'ab') as handler:
    for key, value in Index_Version1_compress.items():
        value = "".join(value)
        handler.write(bytearray(value, 'utf8'))
handler.close()

Index_Version2_compress = {}
for term in Index_Version2_uncompress.keys():
    posting_list = Index_Version2_uncompress[term][1]
    doc_ids = []
    for listt in posting_list:
        doc_ids.append(int(listt[0]))
    doc_ids.sort()
    gaps = []
    gaps.append(doc_ids[0])
    for i in range(1, len(doc_ids)):
        gaps.append(doc_ids[i] - doc_ids[i-1])
    Index_Version2_compress[term] = get_delta_code(gaps)


def get_common_prefix(listt):
    n = len(listt)
    if n == 0:
        return ""
    if n == 1:
        return listt[0]
    # get min of length of first and last element
    end = min(len(listt[0]), len(listt[n-1]))
    i = 0
    while i < end and listt[0][i] == listt[n-1][i]:
        i += 1
    return listt[0][:i]


terms = [*Index_Version2_compress]
terms.sort()
front_encoding_compressed_terms = []
for start in range(0, len(terms), 8):
    end = start+8
    end = min(end, len(terms))
    prefix = get_common_prefix(terms[start:end])
    front_encoding_compressed_terms.append(str(len(terms[start])))
    front_encoding_compressed_terms.append(prefix)
    front_encoding_compressed_terms.append('*')
    front_encoding_compressed_terms.append(str(terms[start][len(prefix):]))
    for i in range(start+1, end):
        front_encoding_compressed_terms.append(str(len(terms[i]) - len(prefix)))
        front_encoding_compressed_terms.append(terms[i][len(prefix):])
        front_encoding_compressed_terms.append('$')


with open("../out/Index_Version2_compress", 'w') as handler:
    handler.write("".join(front_encoding_compressed_terms))
handler.close()

with open("../out/Index_Version2_compress", 'ab') as handler:
    for key, value in Index_Version2_compress.items():
        value = "".join(value)
        handler.write(bytearray(value, 'utf8'))
handler.close()

#########################################################
# Stastistics
#########################################################

print("4. Size of Index_Version1_compress", os.path.getsize("../out/Index_Version1_compress"))
print("5. Size of Index_Version2_compress", os.path.getsize("../out/Index_Version2_compress"))
print("6. Number of postings in Index_Version1", len(Index_Version1_uncompress))
print("7. Number of postings in Index_Version2", len(Index_Version2_uncompress))
print("8. DF, TF, inverted list length (in bytes) in Index")
query8_terms = ['reynolds', 'nasa', 'prandtl', 'flow', 'pressure', 'boundary', 'shock']

print("LEMMATIZED TOKENS")
header = ['TERMS','DF','TF','SIZE OF INVERTED LIST']
t = PrettyTable(header)
for term in query8_terms:
    try:
        posting_list = Index_Version1_uncompress[term]
        df = posting_list[0]
        tf = 0
        for listt in posting_list[1]:
            tf += listt[1]
        list_length = sys.getsizeof(posting_list[1])
        t.add_row([term, str(df), str(tf), str(list_length)])
    except KeyError:
        continue
print(t)
print("STEMMED TOKENS")
t = PrettyTable(header)

for term in query8_terms:
    try:
        posting_list = Index_Version2_uncompress[term]
        df = posting_list[0]
        tf = 0
        for listt in posting_list[1]:
            tf += listt[1]
        list_length = sys.getsizeof(posting_list[1])
        t.add_row([term, str(df), str(tf), str(list_length)])
    except KeyError:
        continue
print(t)

posting_list = Index_Version1_uncompress['nasa']
df = posting_list[0]
listt = posting_list[1]
sorted_d = sorted(listt, key=lambda x: x[0])
print("9. DF of 'nasa' :", df, "First 3 entries in posting list for lemmatized token 'nasa'")
t = PrettyTable(['Term', 'Doc_id', 'TF', 'Max_TF', 'Doc length'])
for i in range(0,3):
    t.add_row(['nasa'] + list(sorted_d[i]))
print(t)

posting_list = Index_Version2_uncompress['nasa']
df = posting_list[0]
listt = posting_list[1]
sorted_d = sorted(listt, key=lambda x: x[0])
print("DF of 'nasa' :", df, "First 3 entries in posting list for stemmed token 'nasa'")
t = PrettyTable(['Term', 'Doc_id', 'TF', 'Max_TF', 'Doc length'])
for i in range(0,3):
    t.add_row(['nasa'] + list(sorted_d[i]))
print(t)

term_df = {}
doc_max_tf = {}
doc_doc_len = {}
for key, value in Index_Version1_uncompress.items():
    term_df[key] = value[0]
    max_tfs = []
    for entry in value[1]:
        if entry[0] in doc_max_tf.keys():
            doc_max_tf[entry[0]] = max(doc_max_tf[entry[0]], entry[2])
        else:
            doc_max_tf[entry[0]] = entry[2]
        if entry[0] in doc_doc_len.keys():
            doc_doc_len[entry[0]] = max(doc_max_tf[entry[0]], entry[3])
        else:
            doc_doc_len[entry[0]] = entry[3]

maximum_key = max(term_df, key=term_df.get)
minimum_key = min(term_df, key=term_df.get)
maximum_keys = []
minimum_keys = []
for key, value in Index_Version1_uncompress.items():
    if value[0] == Index_Version1_uncompress[maximum_key][0]:
        maximum_keys.append(key)
    if value[0] == Index_Version1_uncompress[minimum_key][0]:
        minimum_keys.append(key)

largest_max_tf = max(doc_max_tf, key=doc_max_tf.get)
largest_doc_len = max(doc_doc_len, key=doc_doc_len.get)
print("10. Term from Index_Version1 with largest DF is", maximum_keys)
print("    Number of terms from Index_Version1 with smallest DF is", len(minimum_keys), " and is available in file Index1_smallest_df")
print("12. Document with largest MAX_TF for Index_Version1 is ", largest_max_tf)
print("12. Document with largest Doc_len for Index_Version1 is ", largest_doc_len)

with open("../out/Index1_smallest_df", 'w') as handler:
    handler.write(str(minimum_keys))
handler.close()

term_df = {}
doc_max_tf = {}
doc_doc_len = {}
for key, value in Index_Version2_uncompress.items():
    term_df[key] = value[0]
    max_tfs = []
    for entry in value[1]:
        if entry[0] in doc_max_tf.keys():
            doc_max_tf[entry[0]] = max(doc_max_tf[entry[0]], entry[2])
        else:
            doc_max_tf[entry[0]] = entry[2]
        if entry[0] in doc_doc_len.keys():
            doc_doc_len[entry[0]] = max(doc_max_tf[entry[0]], entry[3])
        else:
            doc_doc_len[entry[0]] = entry[3]
maximum_key = max(term_df, key=term_df.get)
minimum_key = min(term_df, key=term_df.get)
maximum_keys = []
minimum_keys = []
for key, value in Index_Version2_uncompress.items():
    if value[0] == Index_Version2_uncompress[maximum_key][0]:
        maximum_keys.append(key)
    if value[0] == Index_Version2_uncompress[minimum_key][0]:
        minimum_keys.append(key)

largest_max_tf = max(doc_max_tf, key=doc_max_tf.get)
largest_doc_len = max(doc_doc_len, key=doc_doc_len.get)
print("11. Term from Index_Version2 with largest DF is", maximum_keys)
print("    Number of terms from Index_Version2 with smallest DF is", len(minimum_keys), " and is available in file Index2_smallest_df")
print("12. Document with largest MAX_TF for Index_Version2 is ", largest_max_tf)
print("12. Document with largest Doc_len for Index_Version2 is ", largest_doc_len)

with open("../out/Index2_smallest_df", 'w') as handler:
    handler.write(str(minimum_keys))
handler.close()