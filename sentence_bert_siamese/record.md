#### 实验记录

+ 1030-0008: Word2Vec粗排卡75，siamese
+ 1030-0934: train_bert_mean_1_87，Bert粗排卡87，single
+ **1030-1341**: train_bert_mean_1_87，Bert粗排卡87，siamese
+ 1030-2053: train_bert_mean_1_87_add6（卡87增量加7900语料），Bert粗排卡87，siamese
+ 1031-0856: train_bert_mean_1_87_add5（卡85增量加4456语料），Bert粗排卡87，siamese
+ 1031-1959: train_bert_mean_1_87_add7（卡85贪心加100语料），Bert粗排卡87，siamese
+ 1031-2247: train_bert_mean_1_87_add9（卡85贪心加1000语料），Bert粗排卡87，siamese
+ 1031-2258: train_bert_mean_1_87_add8（卡85贪心加500语料），Bert粗排卡87，siamese
+ 1101-1637: train_bert_mean_1_87_add10（卡85贪心加惩罚a-2b加100语料），Bert粗排卡87，siamese
+ 1101-1652: train_bert_mean_1_87_add11（卡85贪心加惩罚a-2b加500语料），Bert粗排卡87，siamese
+ 1101-2358: train_bert_mean_1_87_add12（卡85贪心加惩罚a-b加100语料），Bert粗排卡87，siamese
+ 1102-0942: train_bert_mean_1_87_add15，Bert粗排卡87，siamese
+ 1102-0950: train_bert_mean_1_87_add16，Bert粗排卡87，siamese
+ 1102-0951: train_bert_mean_1_87_add17，Bert粗排卡87，siamese
+ 1102-1713: train_bert_mean_1_87_add18，Bert粗排卡87，siamese
+ 1102-1714: train_bert_mean_1_87_add19，Bert粗排卡87，siamese
+ 1102-1716: train_bert_mean_1_87_add13，Bert粗排卡87，siamese
+ 1103-0004: train_bert_mean_1_87_add14，Bert粗排卡87，siamese
+ 1105-1036: train_bert_mean_1_87，Bert粗排卡87，single
+ 1105-1647: train_bert_mean_1_85_tokenize
+ **1105-1913**: train_bert_mean_1_85_tokenize_add1（加语料时去叠词500），Bert粗排卡87，siamese
+ 1106-1016: train_bert_mean_1_85_tokenize_add2（加语料时去叠词100），Bert粗排卡87，siamese
+ 1106-1655: train_bert_mean_1_85_tokenize_add3（加语料时去叠词），Bert粗排卡87，siamese
+ 1107-0007: train_bert_mean_1_85_tokenize_add4（加语料时去叠词），Bert粗排卡87，siamese
+ 1107-0938: train_bert_mean_1_85_tokenize_add5（加语料时去叠词），Bert粗排卡87，siamese
+ 1108-1206: train_bert_mean_1_85_tokenize_add4（加语料时去叠词）
+ 1108-1314: train_bert_mean_1_85_tokenize_add6（加语料时去叠词）
+ 1108-1515: train_bert_mean_1_85_tokenize_add3（加语料时去叠词）
+ 1114-1810: train_bert_87_triplet_cut20+invalid，triplet
+ 1115-1048: train_bert_87_triplet_cut20+invalid，triplet，margin=1
+ 1116-0007: train_bert_87_triplet_cut20+invalid，triplet
+ 1116-0009: train_bert_87_triplet_cut20+invalid，triplet
+ 1116-1853: train_bert_87_triplet_cut20+invalid，triplet，margin=0.9
+ 1116-1855: train_bert_87_triplet_cut20+invalid，triplet，margin=1.1
+ **1118-1020**: train_bert_87_triplet_cut20+invalid，triplet，margin=0.8
+ 1122-1118: train_bert_87_triplet_cut20+invalid+label，triplet
+ 1122-1857: train_bert_87_triplet_cut20+invalid+label，triplet
+ 1123-1649: train_bert_87_triplet_cut20+invalid+label，triplet
+ 1124-1148: train_bert_87_triplet_cut20+invalid+label，triplet 
+ 1124-1205: train_bert_87_triplet_cut20+invalid+label，triplet 
+ 1125-1510: train_bert_87_triplet_cut20+invalid+label，triplet 
+ 1126-1256: train_bert_mean_1_87.csv，siamese，$<u,v,|u-v|>$，CLS
+ **1127-1653**: train_bert_mean_1_87.csv，siamese，$<u,v,|u-v|>$，MEAN
+ 1128-1051: train_bert_mean_1_87.csv，siamese，$<u,v,|u-v|,u\cdot v>$，CLS
+ 1128-1952: train_bert_mean_1_87.csv，siamese，$<u,v,u\cdot v>$，CLS
+ **1208-2331**: train_bert_mean_1_85_tokenize_add1，siamese ，$<u,v,|u-v|,u\cdot v>$，CLS



#### 添加语料实验记录

| index | add_corpus               | add_count | total_count |
| ----- | ------------------------ | --------- | ----------- |
| 0     | no                       | 0         | 3142        |
| 1     | 全量+0.85                | 14179     | 17321       |
| 2     | 全量+0.85+车贷           | 14243     | 17385       |
| 3     | 全量+0.85+车贷+三新      | 15468     | 18610       |
| 4     | 增量+0.85                | 4436      | 7578        |
| 5     | 增量+0.85+车贷           | 4456      | 7598        |
| 6     | 增量+0.87+车贷           | 7900      | 11042       |
| 7     | 贪心+0.85                | 100       | 3242        |
| 8     | 贪心+0.85                | 500       | 3642        |
| 9     | 贪心+0.85                | 1000      | 4142        |
| 10    | 贪心+0.85+惩罚a-2b       | 100       | 3242        |
| 11    | 贪心+0.85+惩罚a-2b       | 500       | 3642        |
| 12    | 贪心+0.85+惩罚a-b        | 100       | 3242        |
| 13    | 贪心+0.85+惩罚a-b        | 500       | 3642        |
| 14    | 贪心+0.85+惩罚a/(b+5)    | 100       | 3242        |
| 15    | 贪心+0.85+惩罚a/(b+5)    | 500       | 3642        |
| 16    | 贪心+0.85+惩罚a/(b+0.01) | 100       | 3242        |
| 19    | 贪心+0.85+惩罚a/(b+0.01) | 500       | 3642        |
| 17    | 贪心+0.85+惩罚a/(b+1)    | 100       | 3242        |
| 18    | 贪心+0.85+惩罚a/(b+1)    | 500       | 3642        |
| 20    | a-0.2*b > 0              | 0         | 1777        |
| 21    | a > 0                    | 0         | 1902        |
| 22    | a - 0.1*b > -1           | 0         | 2986        |
| 23    | a - 0.2*b > -1           | 0         | 2855        |

| index | add_corpus                         | add_count | total_count |
| ----- | ---------------------------------- | --------- | ----------- |
| 0     | no                                 | 0         | 3142        |
| 1     | bert+tokenize+贪心0.85+惩罚a/(b+5) | 500       | 3642        |
| 2     | bert+tokenize+贪心0.85+惩罚a/(b+5) | 100       | 3242        |
| 4     | bert+tokenize+贪心0.85+惩罚a/(b+5) | 1000      | 4142        |