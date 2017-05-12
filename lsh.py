import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg as LA
from copy import copy
from  itertools import combinations
import  sklearn.metrics.pairwise

import json
import os
os.system('cls')

# load csv data
# the file have 3 columns: URI,name,text for many peopls
wiki = pd.read_csv('people_wiki.csv')
#insert a new column as an indicator
wiki['id'] = np.array([i for i in range(wiki.shape[0])])
# load the tf_idf file
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
tf_idf = csr_matrix( (data, indices, indptr), shape)


#nearest neighbor modelwith Cosine distance as a distance measure,disrtance = 1 - cos(angle between)
start1 = time.time()
model_brute = NearestNeighbors(metric= 'cosine' , algorithm = 'brute')
# use tf_idf as training set
model_brute.fit(tf_idf)
# 10 nearest neighbors to barack with all tfidf as a representation for text

distances, indices = model_brute.kneighbors(tf_idf[35817], n_neighbors = 10)
stop1 = time.time()
time_brute = stop1 - start1
print " it takes brute force search to find knn for one dtp", time_brute
print " to find knn for all dpts it probably will take",time_brute*tf_idf.shape[0]

def lsh_train( data, h = 16, seed = None):
    '''INPUT:
    data: (matrix or np) shape(N,dim)
    h : (integer), number of hyper planes and it determines \
    number of bins = 2^(h)
    seed : (integer) seed for generating random hyper planes
    OUTPUT: is a dictionary consisting of \
           hashtable(dictionary) :bins as key and list of data index belonging
           to each bin as values\
           data_bin_bit (nparray)(N,h) data_bin_bit[i] is the bin that ith data\
           belong to, bin is in the form of h bit encoding
           data_bin_integer(nparray)(N,1) data_bin_integer[i] is the bin that\
           ith data belong to, bin is in the form integer
           h : number of hyperplane, user can either enter it as input or \
           use preset value
           random_vectors : is the equations oh h hyperplanes in dim dimention
    '''
    # 1. construction of h hyper plane in a d dimentional space
    # dimension and here it is # of words = 547979
    dim = data.shape[1]
    # generate samples from standard normal distribution
    # these samples determines the equation of hyperplane in dim dimention space
    if seed is not None:
        np.random.seed(seed)
    random_vectors = np.random.randn(dim, h)

    # 2.for each datapoint,for each hyperplane compute score & translate to binary index
    # you can either use two loops or a much better solution, use matrix operation
    # in other word, put dtp in hyperplane equation
    # each point is in either site of the hyperplane
    # convert T/F to 1/0
    # leading to a h-bit encoding of the bin index
    data_bin_bit = np.array(data.dot(random_vectors) >= 0 , dtype = int )
    # for conviniency, we convert each binary bin index to integer
    powers_of_two = np.array([2**i for i in range(h-1,-1,-1)])
    # result is integer index of bins for all data points
    data_bin_integer = data_bin_bit.dot(powers_of_two)

    # 3. create hash table
    hash_table = {}
    # for every dtps,
    for i in range(len(data_bin_integer)):
        #if the bin that data point is belonging to is already in the hashtable\
        # append the indice of that dtpt to the list of dtpts locating in that bin
        if data_bin_integer[i] in hash_table:
            hash_table[data_bin_integer[i]].append(i)
        # and if this is the first time passing that bin, insert the key of bin\
        # and the indice of that data point to the hashtable
        else :
            hash_table[data_bin_integer[i]] = [i]

    model = { 'hash_table' : hash_table,
             'data_bin_bit' : data_bin_bit,
             'data_bin_integer' : data_bin_integer,
             'random_hyper_planes' : random_vectors,
             'number_hyper_planes' : h}

    return model

start2 = time.time()
model = lsh_train( tf_idf,16, 143)
hash_table = model['hash_table']
data_bin_integer = model['data_bin_integer']
data_bin_bit = model['data_bin_bit']
stop2 = time.time()
time_partitioning = stop2 - start2
print " it took for partitioning ", time_partitioning

def search_nearby_bins( q_bin_bits, hash_table, search_radius=2):
    ''' for a given query and a train_LSH, return all candidates\
    among given search_radius
        INPUT :
            q_bin_bits:bit representation of a bin that q is located in
            hash_table : hash table of a trained LSH
            search_radius : the # of bits that searching neighbor bins differ\
            from the bits of bin query is locating in

        OUTPUT:
               list of all candidates' id for nn search
    '''
    candidates_list = []
    # number of hyperplanes that partition space = nymber of bits
    num_vector = len(q_bin_bits)
    # for changeing bit to integer
    powers_of_two = np.array([2**i for i in range(num_vector-1,-1,-1)])
    # Return tuples of subsequences (search_radius=length of each tuple)\
    #of elements from the input iterable.
    for comb in combinations(range(num_vector), search_radius):
        # make a copy of q_bin_bits, we want to change search_radius of them
        alternate_bits = copy(q_bin_bits)
        # or alternate_bits = np.array([i for i in q_bin_bits])
        # we can iterate over a tuple like a list
        for indice in comb:
            # flip two bits  of a bin that query is in it
            #flip = change 0 to 1 & change 1 to 0
            alternate_bits[ indice ] = 1 - alternate_bits[ indice ]
        nearby_bin_integer = alternate_bits.dot(powers_of_two)
        if nearby_bin_integer in hash_table:
            candidates_list = candidates_list + hash_table[nearby_bin_integer]
    return candidates_list



#main loop
def knn_lsh( max_radius , q, k , model, data = tf_idf):
    '''
    for a given query and a maximun searching area , find knn
    INPUT: max_radius (integer) :max numbers of flipping bits of the bin that
           query stands in, which determines searching areas
           q : (integer)query id
           k: (integer) how many neirest neighbors
           model : model of the partitioned space, we got it from function \
           "lsh_train"
           data : matrix of data_test
    OUTPIT: k nearest neighbors found in the search space(tuple)(distance, id)
            numbers of candidate that we search through (integer)

    '''
    # first Search nearby bins and collect candidates
    all_candidates = []
    #start3 = time.time()
    for rad in range(max_radius+1):
        # for every acceptable radiunse, find candidate points
        candidates_rad = search_nearby_bins\
        ( model['data_bin_bit'][q,:], model['hash_table'], rad)
        # store all candidates'id in a list
        all_candidates = all_candidates + candidates_rad

    #stop3 = time.time()
    #print " takes to find all candidates", (stop3 - start3)
    # remove a query point from list of candidate
    all_candidates.remove(q)
    # pass through every candidate and calculate the distance between query and\
    # a given candidate
    # knn neighbors found so far, first element of tuple is distance, second:id
    # max number of nn is len(all_candidates)
    if k > len(all_candidates):
        k = len(all_candidates)
    # best_k = [(10,0)]*k
    # for can_id in all_candidates:
    #     dis = cosine_distance(data[can_id], data[q])
    #     # if for a candidate, distance is less than the maximum distance of
    #     #all k found so far:
    #     if dis < best_k[k-1][0]:
    #         # replace it with the kth best nns found so for
    #         best_k[k-1] = (dis, can_id)
    #         # sort knn found so far based on distance in a ascending order
    #         # ( the first one is nearest and the last one is furthest to query)
    #         best_k.sort(key=lambda x: x[0])
    # # since we restrict our search to a much smaller points in this method
    # # insted of for loop:
    # #simply calculate distance to all condidates and then sort them $ pick k
    # # use scikit distance
    # dis = sklearn.metrics.pairwise.cosine_distances(data[all_candidates], data[q])
    #start4 = time.time()
    sim = sklearn.metrics.pairwise.cosine_similarity( data[q], data[all_candidates])
    b = sim[0].argsort()[-k:][::-1]
    best_k_sim = sim[0,b]
    best_k_dis = 1 - best_k_sim
    best_k_ind =  [all_candidates[j] for j in b]

    #stop4 = time.time()
    #print " finding knn for all candidate takes", (stop4 - start4)
    return (best_k_dis, best_k_ind, len(all_candidates))

print_result_4 = 0
if print_result_4 ==1:
    (a,b,c) = knn_lsh( 3 , 35817, 10 , model, tf_idf)
    print " number of candidates check within 3 bites : %d" %c

    print " results of nearest neighbors, (distance, id)"
    for i in range(len((a))):
        print "% 15f,% 10d, %20s" %(a[i],b[i], wiki.iloc[b[i]]['name'])

# effect of numeber of bins searching through

time_history =[]
num_candidates_history = []
max_distance_to_query_history = []
min_distance_to_query_history = []
av_distance_to_query_history = []

for max_search_radiunse in range(16):
    start = time.time()
    # find knn by searching through max_search_radiunse given
    (dis, id, num_candidates) = knn_lsh\
    ( max_search_radiunse, q = 35817, k =10, model = model, data =tf_idf)
    stop = time.time()
    # for 10 nn, print their distance to query, their name and id
    print " search_raduinse : ", max_search_radiunse
    for i in range(len((dis))):
        print "% 15f,% 10d, %20s" %(dis[i],id[i], wiki.iloc[id[i]]['name'])

    # neasure time
    time_lentgh = stop - start
    # collect results for differenr search radiunse
    time_history.append(time_lentgh)
    num_candidates_history.append(num_candidates)
    max_distance_to_query_history.append(max(dis))
    min_distance_to_query_history.append(min(dis))
    av_distance_to_query_history.append(sum(dis)/len(dis))



plt.plot(num_candidates_history, linewidth = 4)
plt.ylabel( 'number of candidates searched')
plt.xlabel( 'search radiunse ')

plt.figure()
plt.plot(time_history, linewidth = 4)
plt.ylabel( 'time')
plt.xlabel( 'search radiunse ')

plt.figure()
plt.plot(max_distance_to_query_history, linewidth = 4, label = 'maximum distance to query point')
plt.plot(min_distance_to_query_history, linewidth = 4, label = 'minimum distance to query point')
plt.plot(av_distance_to_query_history, linewidth = 4, label = 'average distance to query point')
plt.ylabel( 'cosine distance to query')
plt.xlabel( 'search radiunse ')
plt.legend(loc ='best')

plt.show()
