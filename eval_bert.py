from collections import defaultdict
import numpy as np
import operator
import sys

def load_nist_qrels():
    rel_dict = defaultdict(list) 
    all_dict = defaultdict(list)
    nonrel_dict = defaultdict(list)

    with open('qrels.robust2004.txt') as pF:
        for line in pF:
            topic, _, doc, label = line.split()
            all_dict[topic].append(doc)
            if int(label) > 0:
                rel_dict[topic].append(doc)
            else:
                nonrel_dict[topic].append(doc)
    return rel_dict, nonrel_dict, all_dict

def load_q_doc_bm25(bm25F):
    doc_dict = {}
    q_dict = {}
    score_dict = {}
    with open(bm25F) as bF:
        for line in bF:
            _, score, q, d, qid, did, qno, dno = line.strip().split('\t')
            doc_dict[dno] = did
            q_dict[qno] = qid
            score_dict[(qno,did.split('_')[0])] = float(score)
    return doc_dict, q_dict, score_dict

def load_q_doc_bert(bertF, doc_dict, q_dict, bm25_dict, topK, w):
    score_dict = defaultdict(dict)
    with open(bertF) as bF:
        for line in bF:
            q, _, d, _, score, _ = line.strip().split()
            sent = doc_dict[d]
            doc = sent.split('_')[0]
            score = float(score)
            if doc not in score_dict[q]:
                score_dict[q][doc] = [score]
            else:
                score_dict[q][doc].append(score)
    for q in score_dict:
        doc_dict = {}
        for d in score_dict[q]:
            scores = score_dict[q][d]
            scores.sort(reverse=True)
            # assert(len(scores) > 5) 
            sum_score = 0
            score_list = []
            weight = 1.0
            rank = 1.0
            for s in scores[:topK]:
            	# sum_score += s
            	score_list.append(s)
            # if len(scores) > 1:
                sum_score += s / rank
                rank += 1
            doc_dict[d] = w * bm25_dict[(q,d)] + (1.0-w) * sum_score

        doc_dict = sorted(doc_dict.items(), key=operator.itemgetter(1), reverse=True)
        rank = 1
        for doc, score in doc_dict:
            print q_dict[q], 'Q0', doc, rank, score, w
            rank+=1

def main():
    topK = int(sys.argv[1])
    w = float(sys.argv[2])
    rel_dict, nonrel_dict, all_dict = load_nist_qrels()
    doc_dict, q_dict, score_dict = load_q_doc_bm25('robust04_bm25_test.csv')
    load_q_doc_bert('prediction.trec_tweet_title', doc_dict, q_dict,
    	score_dict, 2, w)

if __name__ == "__main__":
    main()