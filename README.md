# KNN
fetching similar articles to the article that one is currently reading.
in knn.py, brute force algorithm is used which calculate the distance between all data points and the query point and among all of them, k data points with smallest distances are selected as k nearest neighbores. since data is in the text format, text representation is important and 2 different text represantations are considered. morever, euclidean distance and cosine distance are used as different distance metrics.we use scikit-learn package in this file.

In the lsh.py we use locality sensitive hashing in stead of brute force search. locality sensiitive hashing is a fast, efficient approximate nearesh neighbor search. In this method, we partition space with random hyperplanes and instead of searching through all data points, we start searching in the bin that query point fall in. then the search is continueing in neighboring bins till either computational budget is consumed or the quality of nearest neighbor found is good enough.


