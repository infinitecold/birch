#!/usr/bin/env bash

 # Large-Core17 5 folders alpha
 python eval_bert.py --experiment large_mb_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.4 0 0 0 all
 mv run.large_mb_core17.cv.all run.large_mb_core17.cv.a
 # Large-Core17 5 folders alpha + beta
 python eval_bert.py --experiment large_mb_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.3 0.2 0 0 all
 mv run.large_mb_core17.cv.all run.large_mb_core17.cv.b
 # Large-Core17 5 folders alpha + beta + gamma
 python eval_bert.py --experiment large_mb_core17 --index_path /tuna1/indexes/lucene-index.core17.pos+docvectors+rawdocs --collection core17 --qrels qrels.core17.txt --bm25_res core17_bm25_rm3_cv_sent.txt --folds_path core17-5-folds.json 3 0.5 0.2 0.1 0 all
 mv run.large_mb_core17.cv.all run.large_mb_core17.cv.c