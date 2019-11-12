import sys

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def main():
    dataset = sys.argv[1]
    stop_words = list(stopwords.words('english'))
    with open(dataset, 'r') as in_file, \
            open(dataset.replace('sents.csv', 'pruned_sents.csv'), 'w') as out_file:
        for line in in_file:
            label, score, query, sent, qid, sid, qno, sno = line.strip().split('\t')
            lower_query = filter(lambda t: t not in stop_words, query.lower().split())
            contains = False
            for q in lower_query:
                if q in sent.lower():
                   contains = True
            if not contains:
                out_file.write(line)


if __name__ == '__main__':
    main()
