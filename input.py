import pickle
from spacy.lang.en.stop_words import STOP_WORDS

## Driver program to output the documents related to search query


cluster_info = pickle.load(open("cluster_info.dms","rb"))
keywords_info = pickle.load(open("keywords_cluster_info.dms", "rb"))
stopwords = list(STOP_WORDS)
input_text = input("Enter: ")
#Example input text="is there any vaccination using nano particles"


words = input_text.split()
req_words = []

for word in words:
  if word not in stopwords:
    req_words.append(word)

words_in_clusters = [0] * 5

for word in req_words:
    if word in keywords_info.keys():
        for each_cluster in keywords_info[word]:
          words_in_clusters[each_cluster] += 1

#Find the cluster having max number of words
req_cluster = words_in_clusters.index(max(words_in_clusters))

#Retrive the document IDs corresponding to the above cluster
for each_docID in cluster_info[req_cluster]:
  print("Document ID is "+ each_docID)