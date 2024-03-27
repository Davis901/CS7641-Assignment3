# CS7641-Assignment3
Github: https://github.com/Davis901/CS7641-Assignement2/edit/main/

This project consists of two jupyter notebook processing dimentionality reduction and Clustering algorithm on two different datawet. Notebooks are turned specicifally for datasets: BC->Breast Cancer, BPL-> Bank Peronal Loan.

## imported libararies

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import NearMiss
from sklearn.manifold import TSNE
from time import time
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc, silhouette_score, mean_squared_error, f1_score


## Part 1: Clustering - Kmeans and Expectation Maximization
## Part 2: Dimentionality Reduction - PCA, ICA, RP, TSNE
## Part 3: Clustering with Dimentionality Reduction
## Part 4: NN with Dim. Red.
## Part 5: NN with Clustering Labels.
