########### TEAM MEMBERS ###########
#* Aparna Dutt
#* Pramod Sai Kudapa
#* Anil Rayala
#* Prajwal Chandra

########### BRIEF DESCRIPTION ########
## (i)	First we are downloading CORD-19 dataset from kaggle
## (ii)	From the article jsons we are fetching paper_id and body_text
## (iii)	Generate feature vectors using tf-idf scores
## (iv)	Perform Dimensionality Reduction using PCA
## (v)	Perform K-means Clustering (Tensorflow)
## (vi)	Perform Topic Modeling (LDA) to find important keywords for each cluster

########## LIST OF ALGORITHMS AND SOFTWARE STACK ##########

## Using Spark and Tensorflow as data pipeline
## Regarding concepts/algorithms, we have used TF-IDF,PCA and K-means clustering

import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import json
import glob
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_sci_sm
import string
from pyspark.sql.types import Row
from pyspark.ml.feature import IDF
from pyspark.ml.feature import CountVectorizer as sparkCountVectorizer
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.linalg import DenseVector
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pickle

## Creating SparkContext and SQLContext
sc = SparkContext("local", "cord_19")
sqlContext = SQLContext(sc)

## Helper function to fetch only the desired columns in a json file (article id and text of the article)
def fetch_data(file_path):
	body = []
	with open(file_path) as file:
		content = json.load(file)
		for entry in content['body_text']:
			body.append(entry['text'])
	return (content['paper_id'], ' '.join(body))
	
########### Preprocessing the data	############


punctuations = string.punctuation						# Fetching punctuations from string module
stopwords = list(STOP_WORDS)							# Collecting the common stop_words (in English language) imported from spacy.lang.en.stop_words 

# Extra stop_words which frequently appear in medical articles
custom_stop_words = ['doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
				'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 
				'al.', 'Elsevier', 'PMC', 'CZI', 'www'
				]
# Appending the extra stop words to the eng language stop words
for w in custom_stop_words:
	if w not in stopwords:
		stopwords.append(w)

# Parser for parsing the text in the article
parser = en_core_sci_sm.load(disable=["tagger", "ner"])			# Loading the parse from en_core_sci_sm package
parser.max_length = 3000000

  
## Helper function to tokenize the full text in an article
def spacy_tokenizer(text):
	all_tokens = parser(text)								## Parse the article using parser defined above
	lem_tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in all_tokens ]		##Lemmatization
	filtered_tokens = [ word for word in lem_tokens if word not in stopwords and word not in punctuations ]				## Filtering stop words and punctuations
	tokens = [token for token in filtered_tokens]
	return tokens


########******* Fetching all the json files (these are medical articles in our dataset)	#############*********

root_path = '/content/document_parses'								## Defining base path
all_json_paths = glob.glob(f'{root_path}/**/*.json', recursive=True)		## Fetching all the jsons paths in our dataset (around 85k)


## Spark Code

sample_jsons = all_json_paths[:100]
json_file_paths = sc.parallelize(sample_jsons)						## Creating RDD of 100 jsons as sample
papers = json_file_paths.map(lambda path: fetch_data(path))				## Fetching data from all json paths
processed_papers = papers.map(lambda t: (t[0], spacy_tokenizer(t[1])))	## Processing the data


## Helper function to convert RDD to PYSPARK DataFrame
def row_conversion(tup):
	labels = ['paper_id','body_text']
	temp_dict = {}
	for i in range(len(tup)):
		temp_dict[labels[i]] = tup[i]
	return temp_dict

## Converting RDD to DataFrame

df = processed_papers.map(lambda record: Row(**row_conversion(record))).toDF()
#df.printSchema()

## Featurizing processed text into TF-IDF vectors
cv = sparkCountVectorizer(inputCol = 'body_text',outputCol = 'tf_vector')
cv_model = cv.fit(df)
tf_df = cv_model.transform(df)						## New column tf_vector with respective term-frequency vectors

## Standardizing TF vectors into TF-IDF vectors
idf = IDF(inputCol='tf_vector',outputCol='tfidf_vector')
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)					## New column tfidf_vector with respective TF-IDF vectors

## Helper function to convert sparse vector to dense vector
def sparse_to_dense(v):
	v = DenseVector(v)
	dense_vector = list([float(x) for x in v])
	return dense_vector

## Converting back to RDD
papers_rdd = tfidf_df.select('paper_id', 'tfidf_vector').rdd.map(lambda t: (t['paper_id'], sparse_to_dense(t['tfidf_vector'])))

tfidf_dict = dict(papers_rdd.take(100))				## Considering only 100 papers as sample

sc.stop()		## Stopping SparkContext.Not using it any further

tfidf_matrix = np.array(list(tfidf_dict.values()))		## Converting to numpy matrix (100, ~23k)

## Applying Dimensionality Reduction using PCA
## Reducing dimensions to preserve 95% variance in the original data

pca = PCA(n_components = 0.95)
transformed_input = pca.fit_transform(tfidf_matrix)		## Dimensions reduced from ~23k to 55

## TENSORFLOW TO PERFORM K-MEANS CLUSTERING (k = 5)

## Helper function to convert matrix to tensor
def train_function():
	return tf.compat.v1.train.limit_epochs( tf.convert_to_tensor(transformed_input, dtype=tf.float32), num_epochs=1)

instances, features = transformed_input.shape				## Number of observations and dimensions
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=5)			## Kmeans node

# Training
generations = 10				## Number of iterations
old_centroids = None			## Initialization of centroids
for generation_no in range(generations):
	kmeans.train(train_function)						## Performing Kmeans
	updated_centroids = kmeans.cluster_centers()			## Fetching cluster centers
	old_centroids = updated_centroids					## Updating the old centroids

## Assigning documents to the respective cluster

cluster_labels = list(kmeans.predict_cluster_index(train_function))		## Finding cluster groups
for idx, each_vector in enumerate(transformed_input):
	cluster_idx = cluster_labels[idx]
	document_centroid = updated_centroids[cluster_idx]
	print('Document:', each_vector, 'belongs to ', cluster_idx, ' cluster centered at', document_centroid)

#### PERFORMING LDA ####


count_vectors = []						## Variable for Word Vectors
num_of_clusters = 5
for _ in range(0, num_of_clusters):
  cv = CountVectorizer(stop_words='english', lowercase=True)
  count_vectors.append(cv)
 

#We need to collect all documents belonging to a cluster before using CountVectorizer
cluster_content = {}
for idx in range(0,num_of_clusters):
    cluster_content[idx] = []

for idx, each_vector in enumerate(transformed_input):
    cluster_content[cluster_labels[idx]].append(index_docID_dict[idx][1])		## Grouping articles according to clusters they are part of

count_vectors_output = []
for idx in range(0,num_of_clusters):
    if idx in cluster_content:
        count_vectors_output.append(count_vectors[idx].fit_transform(cluster_content[idx]))			## Applying Count vectoriser on individual clusters

#Performing LDA
numberOfTopics = 15

LDA_models = []
for _ in range(0, num_of_clusters):
    LDA_models.append(LatentDirichletAllocation(n_components=numberOfTopics, max_iter=20, learning_method='online'))		## Constructing 5 LDA models

lda_output = []

for idx in range(0,num_of_clusters):
  lda_cluster_output = LDA_models[idx].fit_transform(count_vectors_output[idx])
  lda_output.append(lda_cluster_output)


## Helper funtion to find keywords from each topic in a cluster
def fetch_keywords(lda_model, cluster_vector, number_of_words):
    keywords_for_cluster = []
    
    for idx, each_topic in enumerate(lda_model.components_):
        keywords_in_each_topic = [(cluster_vector.get_feature_names()[i], each_topic[i]) for i in each_topic.argsort()[:-number_of_words - 1:-1]]		## Topic wise top keywords
        keywords_for_cluster.append(keywords_in_each_topic)

    return keywords_for_cluster

## Finding important keywords for each cluster
keywords_all_clusters = []
for idx, each_model in enumerate(LDA_models):
    if count_vectors[idx] is not None:
      keywords_in_cluster = fetch_keywords(each_model, count_vectors[idx], 4)			## Fetching top 4 keywords for each topic in for individual clusters 
      keywords_all_clusters.append(keywords_in_cluster)

## We will store keywords and cluster info in a file

cluster_info = {}

for idx in range(0,num_of_clusters):
    cluster_info[idx] = []

for idx,_ in enumerate(transformed_input):
    cluster_info[cluster_labels[idx]].append(index_docID_dict[idx][0])

pickle.dump(cluster_info, open("cluster_info","wb"))

## We will make another dictionary with keywords as keys and clusters linked to it as values
keywords_cluster_info = {}

for idx,each_cluster in enumerate(keywords_all_clusters):
    for each_topic in each_cluster:
        for each_keyword in each_topic:
            try:
                keywords_cluster_info[each_keyword[0]].append(idx)
            except KeyError:
                keywords_cluster_info[each_keyword[0]] = [idx]


pickle.dump(keywords_cluster_info, open("keywords_cluster_info","wb"))