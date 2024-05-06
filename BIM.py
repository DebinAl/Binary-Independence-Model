import math
import pandas as pd
from tabulate import tabulate
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def pre_processing(document: str, language_option: str) -> list:
    nltk_stopwords = set(stopwords.words(language_option))

    if language_option == 'indonesian':
        stemmer = StemmerFactory().create_stemmer()
    elif language_option == 'english':
        stemmer = PorterStemmer()

    document = str(document).casefold()     # casefold
    token = word_tokenize(document)         # tokenization

    # stopword and punctuation
    result = [
        word 
        for word in token
        if word.isalnum() 
        and word not in nltk_stopwords        
        ]  

    # stemming
    result = [
        stemmer.stem(word)
        for word in result
    ]

    return result

def inverse_index(query: str, documents: list):
    inverted_index = defaultdict(list)
    query_list = []

    for doc_id, document in enumerate(documents):
        inverted_index[doc_id] = {}
        for term in query:
            if term not in inverted_index:
                if term not in query_list:
                    query_list.append(term)
                tf = document.count(term)
                if tf == 0:
                    continue
                inverted_index[doc_id][term] = tf

    return inverted_index, query_list

def term_weighting(index, query):
    term_weight = {}
    for term in query:
        term_weight[term] = 0

    for doc_id, data in index.items():
        if not data:
            continue
        for term in query:
            if term in data:
                term_weight[term] += data[term]
            else:
                term_weight[term] += 0

    for term, amount in term_weight.items():
        term_weight[term] = amount / len(index)

    return term_weight

def calculate_similarity(Qk, query, index):
    Pk = 0.5
    score = {}
    for doc_id, data in index.items():
        score[doc_id] = 0
        for term in query:
            if term in data:
                score[doc_id] += 1*data[term]*math.log10((Pk*(1-Qk[term]))/(Qk[term]*(1-Pk)))
            elif term not in data:
                score[doc_id] += 1*0*math.log10(Pk)
    return score

def read_documents() -> None:
    query = "penerapan rancangan metode"
    data = pd.read_excel("./src/data.xlsx", usecols=["Judul"])
    documents = data["Judul"].tolist()
    processed_document = []

    processed_query = pre_processing(query, "indonesian")

    for i, document in enumerate(documents):
        processed_document.append(pre_processing(document, "indonesian"))
        print(i+1)
        if i+1 == 150:
            break;

    inverted_index, query_list = inverse_index(processed_query, processed_document)
    term_weight = term_weighting(inverted_index, query_list)
    score = calculate_similarity(term_weight, query_list, inverted_index)
    # print(inverted_index)
    # print(term_weight)

    df = pd.DataFrame.from_dict(score, orient="index", columns=["score"])
    df.to_csv("./result/result.csv")

    sorted_score = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))
    processed_score = {k: v for k, v in sorted_score.items() if v != 0.0}
    
    sorted_df = pd.DataFrame.from_dict(processed_score, orient="index", columns=["score"])
    sorted_df.to_csv("./result/processed.csv")

def main() -> None:
    read_documents()

if __name__ == "__main__":
    main()