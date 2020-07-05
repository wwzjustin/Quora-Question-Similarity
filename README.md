# Quora-Question-Similarity

# 1. Business Problem
## 1.1 Description
Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.
Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.
Credits: Kaggle
Problem Statement
Identify which questions asked on Quora are duplicates of questions that have already been asked. This could be useful to instantly provide answers to questions that have already been answered. We are tasked with predicting whether a pair of questions are duplicates or not.
## 1.2 Sources/Useful Links
Source : https://www.kaggle.com/c/quora-question-pairs
Useful Links
Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments
Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30
## 1.3 Real world/Business Objectives and Constraints
### 1. The cost of a mis-classification can be very high.
### 2. You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice. 
### 3. No strict latency concerns.
### 4. Interpretability is partially important.
     
## 2.1 Data
### 2.1.1 Data Overview
- Data will be in a file Train.csv
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate - Size of Train.csv - 60MB
- Number of rows in Train.csv = 404,290
### 2.1.2 Example Data point
"id","qid1","qid2","question1","question2","is_duplicate"
"0","1","2","What is the step by step guide to invest in share market in india?","What is the s tep by step guide to invest in share market?","0"
"1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Ind ian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
"7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1" "11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comm ents?","1"
2.2 Mapping the real world problem to an ML problem
2.2.1 Type of Machine Leaning Problem
It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.
### 2.2.2 Performance Metric
Source: https://www.kaggle.com/c/quora-question-pairs#evaluation Metric(s):
log-loss : https://www.kaggle.com/wiki/LogarithmicLoss Binary Confusion Matrix
### 2.3 Train and Test Construction
We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.


# 3. Feature selection

## 3.1 Basic Feature Extraction (before cleaning)
f| freq_qid1 = Frequency of qid1's                                                                                                                                                                      |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| freq_qid2 = Frequency of qid2's                                                                                                                                                                      |
| q1len = Length of q1                                                                                                                                                                                 |
| q2len = Length of q2                                                                                                                                                                                 |
| q1_n_words = Number of words in Question 1                                                                                                                                                           |
| q2_n_words = Number of words in Question 2                                                                                                                                                           |
| word_Common = (Number of common unique words in Question 1 and Question 2) word_Total =(Total num of words in Question 1 + Total num of words in Question 2) word_share = (word_common)/(word_Total) |
| freq_q1+freq_q2 = sum total of frequency of qid1 and qid2 freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2                                                                        |
|                                                                                                                                                                                                      |

## 3.2 Advanced Feature Extraction (NLP and Fuzzy Features)

cwc_min : Ratio of common_word_count to min lenghth of word count of Q1 and Q2 cwc_min = common_word_count / (min(len(q1_words), len(q2_words))
cwc_max : Ratio of common_word_count to max lenghth of word count of Q1 and Q2 cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
csc_min : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2 csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
csc_max : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2 csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
ctc_min : Ratio of common_token_count to min lenghth of token count of Q1 and Q2 ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
ctc_max : Ratio of common_token_count to max lenghth of token count of Q1 and Q2 ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
last_word_eq : Check if First word of both questions is equal or not last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])
first_word_eq : Check if First word of both questions is equal or not first_word_eq = int(q1_tokens[0] == q2_tokens[0])
abs_len_diff : Abs. length difference
abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
mean_len : Average Token Length of both Questions mean_len = (len(q1_tokens) + len(q2_tokens))/2
fuzz_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
fuzz_partial_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
token_sort_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
token_set_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
longest_substr_ratio : Ratio of length longest common substring to min lenghth of token count of Q1 and Q2 longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))


# 4. Model result and interpretation

![Image](https://github.com/wwzjustin/Quora-Question-Similarity/blob/master/result.png)


