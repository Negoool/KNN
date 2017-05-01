''' find similar wiki pages as Barak Obama'''

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import  sklearn.metrics.pairwise
import json
from sets import Set
import matplotlib.pyplot as plt

import os
os.system('cls')

# load csv data
# the file have 3 columns: URI,name,text for many peopls
wiki = pd.read_csv('people_wiki.csv')
wiki.info()

#insert a new column as an indicator
wiki['id'] = np.array([i for i in range(wiki.shape[0])])
#print wiki.head()

# load the word count file
loader = np.load('people_wiki_word_count.npz')
#### result is [shape, data, indices, indptr]
# shape = [59071 547979] which is the number of pages(text) and number of words
shape = loader['shape']
# all nonzero elements and len(data) = NNZ = 10379283
data = loader['data']
# columns which NZ element are there so it is between 0,...,547979
# and len(indices) == NNZ
indices = loader['indices']
# it is comulative sum of NZ elements in each raw
# len(indptr) = # texts +1 = 59072
indptr = loader['indptr']

# convert compress sparse raw(csr) to non compress
#  this is a precounted of words obtained from wiki file
# we will work on this matrix in the first part
word_count = csr_matrix( (data, indices, indptr), shape)

# nearest neighbor modelwith Euclidean distance as a distance measure
model = NearestNeighbors(metric= 'euclidean' , algorithm = 'brute')
# use word count(bag of words) as training set
model.fit(word_count)

# query point
print "-"*40
print(" barack obama information")
print wiki[wiki['name'] == 'Barack Obama']

# 10 nearest neighbors to barack with all #of words as a representation for text
distances, indices = model.kneighbors(word_count[35817], n_neighbors = 10)

# show results of knn
# print the name, indice and distance to these nearest 10 points to Barack
print"-"*50
print "10 nearest neigbors to barack obama using bag of word and euclidean distance"
for i in range(10):
    a =  wiki[wiki['id'] == indices[0][i]]
    print "% 10s %15s" %(a['name'] , distances[0][i] )
    #"%15f|%15d |%20s" %(distances[0][i], indices[0][i],a['name'].astype('str'))
print"-"*40


def most_used_word(word_count_id, n_word):
    ''' INPUT: raw of data which is number of counts of words for a given id
        (matrix)
        OUTPUT: (list of tuple) of n most used words and their frequency
    '''
    # since this matris is sparse let's get rid of sparsity
    # ** the function nonzero returns the indexes of nonzero elements in one tuple of two np arrays
    nnz_arg = (np.nonzero(word_count_id))[1]
    # for obama the number of nnz is ~300 which shows too sparsity
    print len(nnz_arg)
    a_list = []
    for arg in nnz_arg :
        # list of tuples[( non-zero argument , # of word for that argument), ( , ) ,...  ]
        a_list.append((arg , word_count_id[0,arg]))
        # rort list based on the secoundelement of tuple which is frequency of words
    a_list.sort(key=lambda tup: tup[1], reverse = True)
    # show the n first ones which have the most frequency
    return  a_list[0:n_word]

# read the file word to index mapping which is a dictionary
with open('people_wiki_map_index_to_word.json') as json_data:
    map_file = json.load(json_data)

def map_map_index_to_word(words_index):
    '''function that maps words index to words
    INPUT : (numpy array) np or list of indexes
    OUTPUT (list) of words for array od indexes:
    '''
    a = []
    for i in range(len(words_index)):
        # search in the dictionary for the same indexed as given words_index
        for word, index in map_file.iteritems():
            if index == words_index[i]:
                # save word if the index match
                a.append(word)
    return a

## explore the most used words in Barack Obama page using those two functions
def print_most_words(word_count_id,number_words, idd):
    #  most used words
    most_words = most_used_word(word_count_id, number_words )
    # mose word indexes
    most_word_index = [x[0] for x in most_words]
    #frequency of most words
    most_word_num = [x[1] for x in most_words]

    # the most used word themselves
    most_words = map_map_index_to_word(most_word_index)
    # print the most used word and number of their appereance
    person = wiki[wiki['id']== idd]
    print person['name']
    print "most used words " + " " *5 + "# of their use"
    for i in xrange(len(most_words)):
        print "%15s: %20d" %(most_words[i], most_word_num[i])

# print most common words in Barack obama page
print_most_words(word_count[35817], 10, 35817)
# print most common words in Francisco Barri page
print "-"*30
print_most_words(word_count[13229], 10, 13229)
print "-"*40

## the 5 most used words in bruck page which exist in barri's pages too
# we convert them to set since intersection can be made easily with sets
def to_set(word_count_id):
    ''' a function that gives a set of existing words for a particular page
    INPUT : words_count_id (raw of sparse matrix for an individual page)
        in other words, it is an array of # of words for an individual page
    OUTPUT : set_existing_words(set) which is the set oh INDEXINGof words
    '''
    # argument of non zero elements, result is a tuple of np array
    nnz_index = np.nonzero(word_count_id)
    # get the secound element o tuple as the index of non zero elements and makes a list of indexes
    set_existing_words = set(nnz_index[1].flatten())
    #return the set of the indexes of those word with # > 1 ( so at least they exist once in a text)
    return set_existing_words

barack_set =   to_set(word_count[35817])
barri_set =  to_set(word_count[13229])
# indexes common words in bruck and barri's pages
common_words = barack_set & barri_set
# print number of common words between obama and barri
print"number of common words between barack and barri : %d" %(len(common_words))
print "\n"
ari1 = []
# creat a list of ( common indexes , frequency of them in obama page)
for member_index in common_words:
    ari1.append(( member_index , word_count[35817,member_index]))
# sort the list based on second element
ari1.sort(key=lambda x: x[1], reverse = True)
#take the five top which has the greatest frequency
top_five = ari1[0:5]
print " five  common words between barack and barri with highest frequency in barack page:"
print top_five
# find the relating words
print map_map_index_to_word([x[0] for x in top_five])
print"\n"

## the number of pages that has these five words
# consider the first word of these 5 wordes
# find all pages(index of pages) that has non zero # of this words
set_intersection = set( (np.nonzero(word_count[:,top_five[0][0]]))[0].flatten() )
for i in range(1,len(top_five)):
    # for the rest four words, find the set of pages that # of that word > 0
    set1 = set( (np.nonzero(word_count[:,top_five[i][0]]))[0].flatten() )
    # each time intersect with the previouse one
    set_intersection = set1 & set_intersection
print "number of pages which has these words :%d" %(len( set_intersection ))

## measuring eacludean distance
word_count_obama = word_count[35817]
word_count_bush = word_count[28447]
word_count_biden = word_count[24478]

dist_obama_bush = sklearn.metrics.pairwise.euclidean_distances(word_count_obama,word_count_bush)
dist_obama_biden = sklearn.metrics.pairwise.euclidean_distances(word_count_obama,word_count_biden)
dist_bush_biden = sklearn.metrics.pairwise.euclidean_distances(word_count_bush,word_count_biden)
# print "dist_obama_bush : %f " %(dist_obama_bush)
# print "dist_obama_biden : %f " %(dist_obama_biden)
# print "dist_bush_biden : %f " %(dist_bush_biden)



''' secound distance metrics : tf_idf
'''
###### load the tf_idf
loader = np.load('people_wiki_tf_idf.npz')
#### result is [shape, data, indices, indptr]
# shape = [59071 547979] which is the number of pages(text) and number of words
shape = loader['shape']
# all nonzero elements and len(data) = NNZ = 10379283
data = loader['data']
# columns which NZ element are there so it is between 0,...,547979
# and len(indices) == NNZ
indices = loader['indices']
# it is comulative sum of NZ elements in each raw
# len(indptr) = # texts +1 = 59072
indptr = loader['indptr']

# convert compress sparse raw(csr) to non compress
#  this is a orecounted of words obtained from wiki file
# we will work on this matrix
tf_idf = csr_matrix( (data, indices, indptr), shape)
# nearest neighbor modelwith Euclidean distance as a distance measure
model_tfidf = NearestNeighbors(metric= 'euclidean' , algorithm = 'brute')
# use tf_idf as training set
model_tfidf.fit(tf_idf)
# 10 nearest neighbors to barack with all tfidf as a representation for text
distances_tfidf, indices_tfidf = model_tfidf.kneighbors(tf_idf[35817], n_neighbors = 10)

# show results of knn
# print the name, indice and distance to these nearest 10 points to Barack
print "-"*50
print "10 nearest neigbors to barack obama using tf-idf and euclidean distance"
for i in range(10):
    a =  wiki[wiki['id'] == indices_tfidf[0][i]]
    print "% 10s %15s" %(a['name'] , distances_tfidf[0][i] )
    #"%15f|%15d |%20s" %(distances_tfidf[0][i], indices_tfidf[0][i],a['name'].astype('str'))

# sort words  in obama and Schilirio' page base on their tf_idf weights
# print most common words in Barack obama page
print "-"*30
print_most_words(tf_idf[35817], 10, 35817)
# print most common words in Phil Schiliro page
print"-"*30
print_most_words(tf_idf[7914], 10, 7914)
print"-"*40

## the 10 most used words in bruck page which exist in Schilirio's pages too
barack_set =   to_set(tf_idf[35817])
schilirio_set =  to_set(tf_idf[7914])
# indexes common words in bruck and barri's pages
common_words = barack_set & schilirio_set
# print number of common words between obama and barri
print"number of common words between these two : %d" %(len(common_words))
print "\n"
ari1 = []
# creat a list of ( common indexes , weights of them in obama page)
for member_index in common_words:
    ari1.append(( member_index , tf_idf[35817,member_index]))
# sort the list based on second element
ari1.sort(key=lambda x: x[1], reverse = True)
#take the five top which has the greatest frequency
top_ten= ari1[0:10]
# find the relating words
print "common words in barack and schilirio pages listed by tf_idf weights in barack page"
print map_map_index_to_word([x[0] for x in top_ten])
print "\n"
top_five = ari1[0:5]

## the number of pages that has these five words
# consider the first word of these 5 wordes
# find all pages(index of pages) that has non zero # of this words
set_intersection = set( (np.nonzero(tf_idf[:,top_five[0][0]]))[0].flatten() )
for i in range(1,len(top_five)):
    # for the rest four words, find the set of pages that # of that word > 0
    set1 = set( (np.nonzero(tf_idf[:,top_five[i][0]]))[0].flatten() )
    # each time intersect with the previouse one
    set_intersection = set1 & set_intersection
print "number of pages which has the top five words :%d" %(len( set_intersection ))
print"-"*40

### distance
# why biden is not in nearest neighbors?
print "why joe biden is not in this list?"
wiki_biden = wiki[wiki['name'] == 'Joe Biden']
print wiki_biden['id']
## = 24478
tfidf_obama = tf_idf[35817]
tfidf_biden = tf_idf[24478]
# compute euclidean distance between obama and biden( it was more then top ten for sure)
dist_obama_biden_tfidf = sklearn.metrics.pairwise.euclidean_distances(tfidf_obama , tfidf_biden)
print "euclidean distance betweeb obama and biden with tfidf weights : %f " %(dist_obama_biden_tfidf)
# make a matrix containing length(numner of words) in each articles
length_article = word_count.sum(axis=1)
distances_100, indices_100 = model_tfidf.kneighbors(tf_idf[35817], n_neighbors = 100)
#length of hundred nearest neighbors of obama
length_100 = length_article[indices_100.flatten()]

# print distribution of documents length if euclidean distance is used
bins1 = np.linspace(0, 1200, 50)
bins2 = np.linspace(0, 1200, 50)
plt.hist(length_article, bins1, normed = True, color ='k' , alpha=0.5, label='entire wikipedia')
plt.hist(length_100,bins2, normed = True, color = 'r', alpha=0.5,\
label='100 nearest neighbor of obama with euclidean distance')
plt.axvline(length_article[35817], color = 'g', linestyle = '--', label= 'length of barack article')
plt.axvline(length_article[24478], color = 'b', linestyle = '--', label= 'length of biden article ')
plt.title(' distribution of documents length ')
plt.xlabel( 'number of words' )
plt.ylabel( ' percentage of documents')
plt.legend(loc='best')

#nearest neighbor modelwith Cosine distance as a distance measure,disrtance = 1 - cos(angle between)
model_2_tfidf = NearestNeighbors(metric= 'cosine' , algorithm = 'brute')
# use tf_idf as training set
model_2_tfidf.fit(tf_idf)
# 10 nearest neighbors to barack with all tfidf as a representation for text
distances_2_tfidf, indices_2_tfidf = model_2_tfidf.kneighbors(tf_idf[35817], n_neighbors = 10)
# show results of knn
# print the name, indice and distance to these nearest 10 points to Barack
print "-"*50
print "10 nearest neigbors to barack obama using tf-idf and 1-cosine as distance metric"
for i in range(10):
    a =  wiki[wiki['id'] == indices_2_tfidf[0][i]]
    print "% 10s %15s" %(a['name'] , distances_2_tfidf[0][i] )

# print distribution of documents length if cosine  is used
distances_100, indices_100 = model_2_tfidf.kneighbors(tf_idf[35817], n_neighbors = 100)
#length of hundred nearest neighbors of obama
length_100 = length_article[indices_100.flatten()]
plt.figure()
bins1 = np.linspace(0, 1200, 50)
bins2 = np.linspace(0, 1200, 50)
plt.hist(length_article, bins1, normed = True, color ='k' , alpha=0.5, label='entire wikipedia')
plt.hist(length_100,bins2, normed = True, color = 'r', alpha=0.5,\
label='100 nearest neighbor of obama with cosine')
plt.axvline(length_article[35817], color = 'g', linestyle = '--', label= 'length of barack article')
plt.axvline(length_article[24478], color = 'b', linestyle = '--', label= 'length of biden article ')
plt.title(' distribution of documents length ')
plt.xlabel( 'number of words' )
plt.ylabel( ' percentage of documents')
plt.legend(loc='best')

plt.show()
