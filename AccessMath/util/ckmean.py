#!/usr/bin/env python

# title           :ckmean.py
# description     :perform convolutional K means on training image pathces
# author          :siyu zhu, Kardo Aziz, Kenny Davila
# date            :Dec 16th, 2013 - May 2017
# version         :0.2
# usage           :
# notes           :
# python_version  : 3.5

# ==============================================================================
# import modules
import numpy as np
import random
#==============================================================================

class ConvolutionalKMeans:
    def __init__(self, data, K, svm=None, verbose=True):
        # Convolutional K-means
        # INPUT:
        # data: matrix each Row is a sample vector
        # k: 	number of total clusters

        # centers: matrix containing center vectors in columns"""
        self.verbose = verbose

        if data.dtype != np.float32 and data.dtype != np.float64:
            raise Exception("K-means Data must be a floating-point type")

        if self.verbose:
            print("Starting Convolutional K-means...")

        # Initialization of D by randomly pick from training data
        if svm is None:
            if verbose:
                print("Initialized Randomly")

            row_idx = random.sample(range(0, len(data)), K)
            centers = data[row_idx, :]
        else:
            if self.verbose:
                print("Initialized from SVMs")

            s_vectors = svm.support_vectors_
            if s_vectors.shape[0] < K:
                raise Exception("Not enough Support Vectors for selected K")

            idx = random.sample(range(0, s_vectors.shape[0]), K)
            centers = s_vectors[idx]
            if self.verbose:
                print("Initial cluster centers shape : ", centers.shape)

        centers = ConvolutionalKMeans.normalize_vectors(centers)

        # print type(D)
        self.data = data
        self.K = K
        self.centers = centers

    def update(self):
        # update center matrix and hot encoding for convolutional k-means
        # INPUT:
        # centers:  initial estimate of center matrix
        # data:     training sample matrix, each column is a sample
        # OUTPUT:
        # centers:  new cluster center matrix after one time update

        new_centers = self.centers.copy()
        #data = self.data
        norms = np.multiply([np.linalg.norm(self.centers, axis=1)], np.array([np.linalg.norm(self.data, axis=1)]).T)
        # arg = cosine similarities
        cos_sim = np.dot(self.data, self.centers.T) / norms

        # maximum similar
        max_cos_sim = np.max(cos_sim, axis=1)
        max_sim_idx = cos_sim.argmax(axis=1)

        # Update centers
        for i in range(0, self.K):
            # get index of those sample that belong to current cluster
            idx = np.where(max_sim_idx == i)

            # get samples that belong to the current cluster
            data_i = self.data[idx[0], :]

            # cluster similarities ...
            sims_i = max_cos_sim[idx]

            # updating cluster centers using a weighted average of the vectors in the cluster
            # each vector is weighted by its current cosine similarity to the current cluster center ...
            if len(sims_i) != 0:
                # weight each vector in the cluster by its cosine similarity to the current center
                multip = data_i.T * sims_i

                # weighted sum vector...
                summ_num = np.sum(multip, axis=1)

                # total weight ...
                summ_den = np.sum(sims_i)

                # final weighted average ...
                new_centers[i, :] = summ_num / summ_den

        # normalize new centers ...
        new_centers = ConvolutionalKMeans.normalize_vectors(new_centers)

        # obtain the total distance between old centers and new centers ...
        distance = self.getCenterDistance(new_centers)

        # set new centers ...
        self.centers = new_centers

        return distance

    @staticmethod
    def normalize_vectors(mat):
        # normalize rows of a matrix, that each row has a magnitude equals 1.
        # INPUT:
        # mat: matrix for processing
        # OUTPUT:
        # mat_new: each row has norm equals 1

        mat_new = mat.copy()
        for i in range(0, mat.shape[0]):
            row = mat[i, :]

            # normalize the vector
            norm_row = np.linalg.norm(row)
            if norm_row > 0.0:
                row /= norm_row
                mat_new[i, :] = row

        return mat_new

    def getCenterDistance(self, new_centers):
        return np.sum(self.centers * new_centers, axis=1).mean()

    def execute(self, max_iterations=1000, min_similarity=1.0):
        last_similarity = 0.0
        iterations = 0

        while iterations < max_iterations and last_similarity < min_similarity:
            last_similarity = self.update()
            iterations += 1

            if self.verbose:
                print("it {0:d}, mean similarity: {1:.10f}".format(iterations, last_similarity))

        return iterations, last_similarity
