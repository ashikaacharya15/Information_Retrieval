import ast, collections, math, os, re, sys
from nltk.stem import WordNetLemmatizer
from prettytable import PrettyTable

base_path = "../../../"
collection_size = 1400


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
    word = word.replace(',', '').replace('/', '').replace('.', '').replace('-', '').replace('\'', '').replace('(', '') \
        .replace(')', '').replace('+', '').replace('*', '')
    return word


stop_words = []
with open(base_path + "Hw1/data/stopwords") as file:
    stop_words = file.read().split()


def read_lemma_index():
    with open(base_path + "hw2/hw2/out/Index_Version1_uncompress.txt") as file:
        content = file.read()
    return ast.literal_eval(content)


def get_queries():
    with open(base_path + "Hw3/RetrievalModel/data/hw3.queries") as file:
        content = file.readlines()
    q = ""
    query = []
    for i in range(0, len(content)):
        if not content[i].startswith("Q"):
            if content[i] == '\n':
                query.append(q)
                q=""
            else:
                q += content[i].replace('\n', ' ')
    query.append(q)
    return query


def parse_query(query):
    query_content = query.split(' ')
    query_tokens = []
    for word in query_content:
        if not is_symbol(word):
            word = remove_special_characters(word)
            if word is not '' and word not in stop_words:
                query_tokens.append(word.lower())
    return query_tokens


def get_lemmas(query_tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = []
    for tokens in query_tokens:
        lemmas.append(wordnet_lemmatizer.lemmatize(tokens, ))
    return lemmas


def get_word_per_document():
    word_per_doc = {}
    for root, dir, files in os.walk(base_path + "Hw1/data/cranfield"):
        for file in files:
            with open(base_path + "Hw1/data/cranfield/" + file) as f1:
                content = f1.read().split()
                word_per_doc[file.split('d')[1]] = len(file_preprocess(content))
    return word_per_doc


def compute_weight(weight_function, tf, max_tf, df, doc_len, avg_doc_len):
    if weight_function == "w1":
        return ((0.4 + (0.6 * math.log10(tf+0.5) / math.log10(max_tf+1))) *
                (math.log10(collection_size/df) / math.log10(collection_size)))

    return ((0.4 + 0.6 * (tf / (tf + 0.5 + 1.5 * (doc_len / avg_doc_len)))) *
            (math.log10(collection_size/df) / math.log10(collection_size)))


def get_document_rankings(weight_function, query_lemmas, lemma_index, word_per_doc):
    score = {}
    query_weights = {}

    for term in query_lemmas.keys():
        if term in lemma_index.keys():
            df = lemma_index[term][0]
            doc_len, avg_doc_len = 0, 0

            if weight_function == "w2":
                doc_len = len(query_lemmas)
                avg_doc_len = sum(word_per_doc.values()) / len(word_per_doc)

            w_t_q = compute_weight(weight_function, query_lemmas[term], max(query_lemmas.values()), df, doc_len, avg_doc_len)
            query_weights[term] = w_t_q
            posting_lists = lemma_index[term][1]

            for item in posting_lists:
                if weight_function == "w2":
                    doc_len = word_per_doc[item[0]]
                    avg_doc_len = sum(word_per_doc.values()) / len(word_per_doc)
                w_t_d = compute_weight(weight_function, item[1], item[2], df, doc_len, avg_doc_len)
                key = str(item[0])
                if key in score.keys():
                    score[key] += w_t_d * w_t_q
                else:
                    score[key] = w_t_d * w_t_q
    return score, query_weights


def get_top5(document_score):
    i = 1
    top_5 = {}
    for key, value in sorted(document_score.items(), key=lambda item: item[1], reverse=True):
        if i <= 5:
            i += 1
            top_5[key] = value
    return top_5


def file_preprocess(content):
    raw_tokens = []
    for i in content:
        if not is_tag(i) and not is_symbol(i):
            i = remove_special_characters(i)
            if i is not '':
                if i not in stop_words:
                    raw_tokens.append(i.lower())
    return raw_tokens


def get_document_weights(weight_function, document_ids, lemma_index, word_per_doc):
    all_document_weights = {}
    headlines = {}
    for root, dir, files in os.walk(base_path + "Hw1/data/cranfield"):
        for file in files:
            file_id = file.split('d')[1]
            if file_id in document_ids:
                doc_weight = {}
                with open(base_path + "Hw1/data/cranfield/" + file) as f1:
                    content = f1.read().split()
                    f1.seek(0)
                    headline = "".join(f1.readlines()).replace('\n', ' ')
                    headlines[file_id] = re.search('<TITLE>(.*)</TITLE>', headline).group(1)[0:150]
                    document_lemmas = collections.Counter(get_lemmas(file_preprocess(content)))
                    for lemma in document_lemmas.keys():
                        doc_weight[lemma] = compute_weight(weight_function, document_lemmas[lemma],
                                                           max(document_lemmas.values()), lemma_index[lemma][0],
                                                           word_per_doc[file_id], (sum(word_per_doc.values()) / len(word_per_doc)))
                all_document_weights[file_id] = doc_weight
    return all_document_weights, headlines


def pretty_print_vector(query_weights):
    header = ["words", "weights"]
    table = PrettyTable(header)
    for word, score in query_weights.items():
        table.add_row([word, score])
    print(table)


def pretty_print_top5(top_5, headlines):
    header = ["Rank", "score", "document id", "headline"]
    table = PrettyTable(header)
    rank = 1
    for doc_id, score in top_5.items():
        table.add_row([rank, score, doc_id, headlines[doc_id]])
        rank += 1
    print(table)


if __name__ == "__main__":
    lemma_index = read_lemma_index()
    queries = get_queries()
    word_per_doc = get_word_per_document()
    for i in [1, 2]:
        filename = "output_w"+str(i)+".txt"
        sys.stdout = open(filename, "w")
        j = 1
        for query in queries:
            print("===================================================================================================")
            print("QUERY", j, ":", query)
            j += 1
            print("===================================================================================================")

            query_tokens = parse_query(query)
            query_lemmas = collections.Counter(get_lemmas(query_tokens))

            print("RESULTS WITH W",i, " WEIGHTING FUNCTION")
            print("===================================================================================================")
            score, query_weights = get_document_rankings("w"+str(i), query_lemmas, lemma_index, word_per_doc)

            print("                     Vector Representation of the query!")
            print("===================================================================================================")
            pretty_print_vector(query_weights)
            print("===================================================================================================")

            top_5 = get_top5(score)
            document_weights, headlines = get_document_weights("w"+str(i), top_5, lemma_index, word_per_doc)
            print("                     Top 5 documents for the query!")
            print("===================================================================================================")
            pretty_print_top5(top_5, headlines)
            print("===================================================================================================")

            print("                     Vector Representation of the TOP 5 documents!")
            print("===================================================================================================")
            for doc, weights in document_weights.items():
                print("Document ID ", doc)
                print(weights)
                print("")
                print("===============================================================================================")
        sys.stdout.close()

