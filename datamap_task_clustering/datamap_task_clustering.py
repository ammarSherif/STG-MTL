"""An implementation for task clustering based on datamaps with Fuzzy-K-Means"""

# ==============================================================================
# Author: Ammar Sherif <ammarsherif90 [at] gmail [dot] com>
# License: MIT
# ==============================================================================

# ==============================================================================
# The file  includes the  code for clustering the  tasks based on  the generated
# datamaps in addition to the code that  pre-process these datamaps. As well, It
# includes some related utilities like plotting the  membership participation to
# each cluster
# ==============================================================================

import enum
from ..fuzzy_kmeans.fuzzy_kmeans.fuzzy_kmeans import FuzzyKMeans
from sklearn.cluster import KMeans
from torch.nn.functional import one_hot
import torch
import numpy as np
import matplotlib.pylab as plt
import copy

# ==============================================================================
# Define an enum to help clarifying the mechanism of clustering,  and whether it
# is point-based or task based.
# ==============================================================================
class ClusteringType(enum.Enum):
    """
    This enumerates the different clustering types

    ----------------------------------------------------------------------------
    Options:
        - POINT_BASED: to cluster the points  of the  datamaps, and use  them as
            votes to calculate the weights of particular cluster
        - TASK_BASED_FUZZY_ITERATIVE: reshapes  the datamap  into vectors; then,
            cluster these vectors using the iterative kmeans
        - TASK_BASED_FUZZY_FROM_KMEANS: reshapes the datamap into vectors; then,
            cluster them using kmeans, and use  the resulting centroids  to cal-
            ulate the membership
        - TASK_BASED_KMEANS: hard clustering with kmeans
    """
    POINT_BASED = 0
    TASK_BASED_FUZZY_ITERATIVE = 1
    TASK_BASED_FUZZY_FROM_KMEANS = 2
    TASK_BASED_KMEANS = 3

# ==============================================================================
# Create our clustering class
# ==============================================================================
class DatamapTaskClustering():
    """
    The class cluster  some tasks  based on the generated  datamaps according to
    different mechanisms:

    - POINT_BASED: will cluster each data point;  then,  the task  membership is
            calculated as the  percentage of number of  points belonging to each
            cluster
    - TASK_BASED_FUZZY_ITERATIVE: clusters  the  whole  data  of  each task into
            different clusters using the Fuzzy KMeans implementation.
    - TASK_BASED_FUZZY_FROM_KMEANS: clusters  the  whole data  of each task into
            different clusters using KMeans; then,  use the  resulting centroids
            to calculate the membership using the fuzzification equation.
    - TASK_BASED_KMEANS: hard clustering with kmeans
    ----------------------------------------------------------------------------
    Inputs:
        - data_map: tensor structure with shape of
                (#tasks, #epochs, #data_points, 2 [std,confidence])
        - n_clusters: number of clusters
        - clustering_type: either POINT_BASED, or TASK_BASED
        - task_names: list of task names [None generates task numbers]
    """
    def __init__(self, data_map, n_clusters,
                 clustering_type = ClusteringType.TASK_BASED_FUZZY_FROM_KMEANS,
                 task_names=None):
        # ----------------------------------------------------------------------
        # Move the datamap to the cpu to avoid any issues later on
        # ----------------------------------------------------------------------
        self.__data_map = data_map.to('cpu')
        self.__clustering_type = clustering_type
        self.__task_names = task_names
        self.__n_clusters = n_clusters
        self.__n_tasks = self.__data_map.shape[0]
        self.__n_data_points = self.__data_map.shape[1]*self.__data_map.shape[2]
        # ----------------------------------------------------------------------
        # Gnerate names as numbers
        # ----------------------------------------------------------------------
        if self.__task_names is None:
            self.__task_names =['Task '+str(i+1) for i in range(self.__n_tasks)]
        # ----------------------------------------------------------------------
        # define the membership tensor of shape (#tasks,#clusters)
        # ----------------------------------------------------------------------
        self.__task_membership = None

    # --------------------------------------------------------------------------

    def __preprocess(self):
        """The method preprocess the datamap before feeding it to our algorithm.
        """

        # ----------------------------------------------------------------------
        # If it is task based, reshape the datamap into (#tasks, data)
        # ----------------------------------------------------------------------
        if self.__clustering_type==ClusteringType.TASK_BASED_FUZZY_ITERATIVE or\
            self.__clustering_type==ClusteringType.TASK_BASED_FUZZY_FROM_KMEANS\
            or self.__clustering_type==ClusteringType.TASK_BASED_KMEANS:
            self.__data_map=self.__data_map.reshape((self.__n_tasks,-1))
        # ----------------------------------------------------------------------
        # If it is point based
        # ----------------------------------------------------------------------
        elif self.__clustering_type == ClusteringType.POINT_BASED:
            self.__data_map=self.__data_map.reshape((-1, 2))

    # --------------------------------------------------------------------------

    def cluster(self, fuzziness_index:float = 1.07, eps:float=0.01,
                max_iter:int=300):
        """
        The method clusters the datamap  and generates a  membership matrix with
        shape of (#tasks, #clusters)
        ------------------------------------------------------------------------
        Inputs:
            - fuzziness_index: if you are using  task clustering with fuzzy k-
                means [default = 1.07]
            - eps: the tolerance value, which is used if we are using  the iter-
                tive fuzzy kmeans [default = 0.01]
            - max_iter: the maximum number of iterations, used with the iterative
                version [default = 300]
        """
        self.__preprocess()
        
        if self.__clustering_type ==ClusteringType.TASK_BASED_FUZZY_FROM_KMEANS:
            kmeans_cluster_estimator = KMeans(n_clusters= self.__n_clusters,
                                              max_iter=max_iter)
            kmeans_cluster_estimator.fit(self.__data_map)
            kmeans_centers = kmeans_cluster_estimator.cluster_centers_

            cluster_estimator = FuzzyKMeans(m = fuzziness_index,
                                            eps=eps,
                                            max_iter=max_iter,
                                            n_clusters= self.__n_clusters)
            self.__task_membership = cluster_estimator.compute_membership(
                    self.__data_map,kmeans_centers)
            return self.__task_membership
        
        elif self.__clustering_type ==ClusteringType.TASK_BASED_FUZZY_ITERATIVE:
            cluster_estimator = FuzzyKMeans(m = fuzziness_index,
                                            eps=eps,
                                            n_clusters= self.__n_clusters)
            cluster_estimator.fit(self.__data_map)
            self.__task_membership = cluster_estimator.fmm_
            return self.__task_membership
        elif self.__clustering_type ==ClusteringType.TASK_BASED_KMEANS:
            kmeans_cluster_estimator = KMeans(n_clusters= self.__n_clusters,
                                              max_iter=max_iter)
            kmeans_cluster_estimator.fit(self.__data_map)
            l = kmeans_cluster_estimator.labels_
            l = torch.from_numpy(l).to(torch.long)
            self.__task_membership = one_hot(l)
            return self.__task_membership
        elif self.__clustering_type == ClusteringType.POINT_BASED:
            cluster_estimator = KMeans(n_clusters=self.__n_clusters)
            cluster_estimator.fit(self.__data_map)
            l = cluster_estimator.labels_
            l = torch.from_numpy(l).to(torch.long)
            labels = one_hot(l)
            res = torch.zeros((self.__n_tasks, self.__n_clusters))
            for c in range(self.__n_tasks):
                res[c,:] = labels[c*self.__n_data_points:(c+1)*\
                                  self.__n_data_points].sum(dim=0)

            totals = res.sum(dim=-1)
            self.__task_membership = torch.transpose(torch.transpose(res,0,
                                                                  1)/totals,0,1)

    # --------------------------------------------------------------------------

    def get_task_membership(self):
        """The method returns the task membership"""
        return self.__task_membership

    # --------------------------------------------------------------------------

    def plot_membership(self, transpose =False, title = None, save_path = None,
                        t_membership=None):
        """
        This method plots the membership of each task to the cluster
        ------------------------------------------------------------------------
        Inputs:
            - transpose: indicates whether to transpose the plot or not
            - title: an optional plot title
            - save_path: an optional path to save the plot
            - t_membership: optional custome task membership
        """
        plt.figure(figsize=(10,10))
        if t_membership is None:
            task_membership = copy.deepcopy(self.__task_membership)
        else:
            task_membership = copy.deepcopy(t_membership)
        num_cluster = task_membership.shape[1]
        xlabels = np.arange(num_cluster)
        ylabels = self.__task_names
        # ----------------------------------------------------------------------
        # transpose the matrix, if needed
        # ----------------------------------------------------------------------
        if transpose:
            task_membership = task_membership.transpose()
            ylabels = np.arange(num_cluster)
            xlabels = self.__task_names

        xlabel_num = task_membership.shape[1]
        ylabel_num = task_membership.shape[0]

        plt.imshow(task_membership)
        # ----------------------------------------------------------------------
        # If title is none, generate one
        # ----------------------------------------------------------------------
        if title is None:
            title = "Membership Matrix ("+str(num_cluster)+") Clusters"

        plt.title(title)

        if not transpose:
            plt.xlabel("Cluster")
            plt.ylabel("Task")
        else:
            plt.xlabel("Task")
            plt.ylabel("Cluster")
        if not transpose:
            plt.xticks(np.arange(xlabel_num),xlabels,size='small')
        else:
            plt.xticks(np.arange(xlabel_num),xlabels,size='small',rotation=45)
        
        plt.yticks(np.arange(ylabel_num),ylabels,size='small')

        # ----------------------------------------------------------------------
        # Show the annotations
        # ----------------------------------------------------------------------
        mid_val = (task_membership.max()+task_membership.min())/2
        for r in range(ylabel_num):
            for c in range(xlabel_num):
                colr = "w"
                if task_membership[r,c] > mid_val:
                    colr = "k"
                plt.annotate('{0:.2f}'.format(task_membership[r,c]),xy=(c,r),
                            ha="center", va="center", color=colr)
        # ----------------------------------------------------------------------
        # save it if needed
        # ----------------------------------------------------------------------
        if save_path is not None:
            plt.savefig(save_path + title + '.png',dpi=200)
        plt.show()
