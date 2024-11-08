#MASH (Manifold Alignment via Stochastic Hopping)

#Import the needed libraries
import graphtools
import numpy as np
from pandas import Categorical
import seaborn as sns
from .vne import find_optimal_t #TODO: Check this out
from itertools import takewhile
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform, _METRICS
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from time import time
from Helpers.rfgap import RFGAP


class MASH: #Manifold Alignment with Diffusion
    def __init__(self, t = -1, knn = 5, distance_measure_A = "default", distance_measure_B = "default", DTM = "log",
                 page_rank = "None", IDC = 1, density_normalization = False,
                 verbose = 0, **kwargs):
        """
    Initialize the MASH class.

    Parameters
    ----------
    t : int or str, optional
        The power to which we want to raise our diffusion matrix. If set to 
        -1 or any string, MASH will find the optimal t value.
    knn : int, optional
        Represents the number of nearest neighbors to construct the graphs.
    distance_measure_A : str or callable, optional
        Either a function, "default", "precomputed" or SciKit-learn metric strings for domain A. If it is a function, 
        it should be formatted like my_func(data) and return a distance measure between points. If set to "precomputed", 
        no transformation will occur, and it will apply the data to the graph construction as given. The graph function 
        compute the distances, but this may be manually changed to use the computed distances unchanged by through kwargs
        (precomputed = "distances"). If set to "default", it will use the graph-created kernels.
    distance_measure_B : str or callable, optional
        Either a function, "default", "precomputed" or SciKit-learn metric strings for domain B. If it is a function, 
        it should be formatted like my_func(data) and return a distance measure between points. If set to "precomputed", 
        no transformation will occur, and it will apply the data to the graph construction as given. The graph function 
        compute the distances, but this may be manually changed to use the computed distances unchanged by through kwargs
        (precomputed = "distances"). If set to "default", it will use the graph-created kernels.
    page_rank : str, optional
        Determines if we want to apply Page Ranking or not. 'off-diagonal' means we only want to apply the Page Ranking 
        algorithm to the off-diagonal matrices, and 'full' means we want to apply the Page Ranking algorithm across the 
        entire block matrix.
    IDC : float, optional
        Stands for Inter-domain correspondence. It is the similarity value for anchor points between domains. Often, it 
        makes sense to set it to be maximal (IDC = 1), although in cases where the assumptions (1: the corresponding points 
        serve as alternative representations of themselves in the co-domain, and 2: nearby points in one domain should remain 
        close in the other domain) are deemed too strong, the user may choose to assign IDC < 1.
    density_normalization : bool, optional
        If set to True, it will apply a density normalization to the joined domains.
    DTM : str, optional
        Diffusion Transformation method. Can be set to "hellinger", "kl" or "log".
    **kwargs : dict, optional
        Keyword arguments for graphtools.Graph function.
    """

        #Store the needed information
        self.t = t
        self.knn = knn
        self.page_rank = page_rank.lower()
        self.normalize_density = density_normalization
        self.DTM = DTM.lower()
        self.distance_measure_A = distance_measure_A
        self.distance_measure_B = distance_measure_B
        self.verbose = verbose
        self.kwargs = kwargs
        self.IDC = IDC

        #Ensure there is a random state
        if "random_state" not in self.kwargs.keys():
            self.kwargs["random_state"] = 42

        #This is done for future plotting, and so we don't have to recalculate the embedding each time we want to plot something.
        self.emb = None
    
    def fit(self, dataA, dataB, known_anchors):
        """
        Parameters:
            dataA (np.ndarray): The first domain (or data set).
            dataB (np.ndarray): The second domain (or data set).
            known_anchors (np.ndarray): An array shaped (n, 2) where n is the number of
                corresponding nodes. The first index should be the data point from dataA
                that corresponds to dataB.
        """

        #Print timing data
        if self.verbose > 2:
           print("Time Data Below")

        #Add the data. It will later be normalized.
        self.dataA = dataA
        self.dataB = dataB

        #This stores the length of the datasets A and B
        self.len_A = len(self.dataA)
        self.len_B = len(self.dataB)

        #Build graphs and kernals
        self.build_graphs()

        self.known_anchors = known_anchors
            
        #Change known_anchors to correspond to off diagonal matricies
        self.known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

        #Connect the graphs
        self.print_time()
        self.graphAB = self.merge_graphs()
        self.print_time(" Time it took to compute merge_graphs function:  ")
        
        #Get Similarity matrix
        self.print_time()
        self.similarity_matrix = self.get_similarity_matrix(self.graphAB)
        self.print_time(" Time it took to compute similarity_matrix function:  ")

        #Get Diffusion Matrix. int_diff_dist stands for the integrated diffusion distance.
        self.print_time()
        self.int_diff_dist, self.projectionAB, self.projectionBA = self.get_diffusion(self.similarity_matrix)
        self.print_time(" Time it took to compute diffusion process:  ")

        if self.verbose > 0:
            print("Fit process finished. We recommend calling optimize_by_creating_connections.")

    """                                     <><><><><><><><><><><><><><><><><><><><>     
                                                     HELPER FUNCTIONS BELOW
                                            <><><><><><><><><><><><><><><><><><><><>                                                    """
    
    def print_time(self, print_statement =  ""):        
        """
        Times the algorithms and prints how long it has been since the function was last called.

        Parameters:
            print_statement (str): A statement to print before the timing information.
        """

        #Only do this if the verbose is higher than 3
        if self.verbose > 2:

            #Start time. 
            if not hasattr(self, 'start_time'):
                self.start_time = time()

            #Check to see if it equals None
            elif self.start_time == None:
                self.start_time = time()

            else:
                #We need to end the time
                end_time = time()

                #Create a string to return
                time_string = str(round(end_time - self.start_time, 5))

                #Reset the start time
                self.start_time = None

                print(print_statement + time_string)
    
    def normalize_0_to_1(self, value):
        return (value - value.min()) / (value.max() - value.min())
    
    def build_graphs(self):
        """
        Builds the graph objects and kernels for domains A and B.
        """

        #------------------------    Build dependencies for domain A    ------------------------    
        if self.distance_measure_A != "default":

            #Create kernals
            self.print_time()
            self.kernalsA = self.get_SGDM(self.dataA, self.distance_measure_A)
            self.print_time(" Time it took to execute SGDM for domain A:  ")

            #Create Graphs using our precomputed kernals
            self.print_time()
            self.graph_a = graphtools.Graph(self.kernalsA, knn = self.knn, knn_max = self.knn, precomputed="distance", **self.kwargs)
            self.print_time(" Time it took to execute the graph for domain A:  ")

        else:
            #Create Graphs and allow it to use the normal data
            self.print_time()
            self.dataA = self.normalize_0_to_1(self.dataA)
            self.graph_a = graphtools.Graph(self.dataA, knn = self.knn, knn_max = self.knn, **self.kwargs)
            self.print_time(" Time it took to execute the graph for domain A:  ")

            #Get the Kernal Data from the graphs
            self.print_time()
            self.kernalsA  = np.array(self.graph_a.K.toarray())
            self.print_time(" Time it took to compute kernal A:  ")

        #------------------------    Build dependencies for domain B    ------------------------   
        if self.distance_measure_B != "default":

            #Create kernals
            self.print_time()
            self.kernalsB = self.get_SGDM(self.dataB, self.distance_measure_B)
            self.print_time(" Time it took to execute SGDM for domain B:  ")

            #Create Graphs using our precomputed kernals
            self.print_time()
            self.graph_b = graphtools.Graph(self.kernalsB, knn = self.knn, knn_max = self.knn, precomputed="distance", **self.kwargs)
            self.print_time(" Time it took to execute the graph for domain B:  ")

        else:
            #Create Graphs and allow it to use the normal data
            self.print_time()
            self.dataB = self.normalize_0_to_1(self.dataB)
            self.graph_b = graphtools.Graph(self.dataB, knn = self.knn, knn_max = self.knn,  **self.kwargs)
            self.print_time(" Time it took to execute the graph for domain B:  ")

            #Get the Kernal Data from the graphs
            self.print_time()
            self.kernalsB  = np.array(self.graph_b.K.toarray())
            self.print_time(" Time it took to compute kernal B:  ")
    
    def apply_aggregation(self, matrix):
        """
        Apply the aggregation function to a powered diffusion operator.

        Parameters:
            matrix (np.ndarray): The matrix to which the aggregation function will be applied.

        Returns:
            np.ndarray: The aggregated matrix.
        """
        
        #The Hellinger algorithm requires that the matricies have the same shape
        if self.DTM == "hellinger" and self.len_A == self.len_B:
            #Apply the hellinger process
            agg_matix = self.hellinger_distance_matrix(matrix)

        elif self.DTM == "kl" and self.len_A == self.len_B:
            #Apply the hellinger process
            agg_matix = self.kl_divergence_matrix(matrix)

        else:
            if (self.DTM == "hellinger" or self.DTM == "kl") and self.verbose > 0:
                print("Unable to compute hellinger or kl because datasets are not the same size.")

            #Squareform it :) --> TODO: Test the -np.log to see if that helps or not... we can see if we can use sqrt and nothing as well. :)
            agg_matix = (squareform(pdist((-np.log(0.00001+matrix))))) #We can drop the -log and the 0.00001, but we seem to like it
    
            #Normalize the matrix
            agg_matix = self.normalize_0_to_1(agg_matix)

        return agg_matix

    def get_SGDM(self, data, distance_measure):
        """
        SGDM - Same Graph Distance Matrix.
        This returns the normalized distances within each domain.

        Parameters:
            data (np.ndarray): The data for which the distance matrix is computed.
            distance_measure (str or callable): The distance measure to use. Can be a string for predefined metrics or a callable function.

        Returns:
            np.ndarray: The normalized distance matrix.
        """


        #Check to see if it is a function
        if callable(distance_measure):
            return distance_measure(self, data)

        #If the distances are precomputed, return the data. 
        elif distance_measure.lower() == "precomputed":
            return data
        
        #Euclidian and other sci-kit learn methods
        elif distance_measure.lower() in _METRICS:

            #Check to make sure we have no NaNs. If we do, we will change the algorihm
            if np.isnan(data).any(): #NOTE: Test ignoring infinites as well

                if self.verbose > 0:
                    print("NaN's dectected. Calculating distances by ignoring NaN positions and normalizing. May take longer.")

                #Proceed with the NanN adjustments by creating a custom nan function we can pass into pdist
                def nan_metric(row_a, row_b, metric):

                    # Mask for valid (non-NaN) entries
                    valid_mask = ~np.isnan(row_a) & ~np.isnan(row_b)

                    if np.sum(valid_mask) == 0:
                        return np.inf  # If no valid entries, return inf.
                
                    # Calculate the distance using the specified metric only on valid entries
                    dist = metric(row_a[valid_mask], row_b[valid_mask])

                    # Normalize by the number of valid entries
                    return dist / np.sum(valid_mask) #NOTE: This will create bias for Euclidean distance
                
                dists = squareform(pdist(data, metric = lambda u, v: nan_metric(u, v, metric = _METRICS[distance_measure.lower()].dist_func)))
                
            else:
                #Just using a normal distance matrix without Igraph
                dists = squareform(pdist(data, metric = distance_measure.lower())) #Add it here -> if its in already for additionally block

        else:
            raise RuntimeError(f"Did not understand {distance_measure}. Please provide a function, or use strings provided by sk-learn metrics or type 'precomputed'.")

        #Normalize it and return the data
        return self.normalize_0_to_1(dists)
        
    def row_normalize_matrix(self, matrix):
        """
        Returns a row-normalized matrix.

        Parameters:
            matrix (np.ndarray): The matrix to be row-normalized.

        Returns:
            np.ndarray: The row-normalized matrix.
        """

        #Get the sum for each row
        row_sums = matrix.sum(axis=1)

        #Prefrom the row-normalized division
        return matrix / row_sums[:, np.newaxis]

    def apply_page_rank(self, matrix, alpha = 0.99):
        """
        Applies the PageRank modifications to the normalized matrix.

        Parameters:
            matrix (np.ndarray): The row-normalized adjacency matrix.
            alpha (float): The alpha value for the PageRank algorithm. Default is 0.99.

        Returns:
            np.ndarray: The modified matrix incorporating the damping factor and teleportation.
        """


        #Get the shape
        N, M = matrix.shape

        # Apply the damping factor and add the teleportation matrix
        return alpha * matrix + (1 - alpha) * np.ones((N, M)) / N

    def get_similarity_matrix(self, matrix):
        """
        Applies adjustments to get the similarity matrix.

        Returns:
            numpy.ndarray: The similarity matrix.
        """

        #Apply density normalization
        if self.normalize_density:
            matrix = self.density_normalized_kernel(matrix)
        
        #Normalize the matrix
        matrix = self.normalize_0_to_1(matrix)

        return matrix

    def kl_divergence_matrix(self, matrix):
        """
        Calculate the KL divergence matrix between rows of two matrices in a vectorized manner.

        Parameters:
            matrix (numpy.ndarray): This should be the diffused matrix.

        Returns:
            numpy.ndarray: Divergence matrix.
        """

        # Ensure there are no zero values to avoid division by zero
        matrix = np.where(matrix == 0, 1e-10, matrix)

        #Normalize and do all the math to preform the KL divergence. #NOTE -> This likely is sucking up a ton of memory
        matrix = self.normalize_0_to_1(squareform(pdist(np.sum(matrix[:, np.newaxis, :] * np.log(matrix[:, np.newaxis, :] / matrix[np.newaxis, :, :]), axis=2))))

        #Return the block matrix!
        return matrix 
    
    def density_normalized_kernel(self, K):
        """
        Compute the density-normalized kernel matrix.

        Parameters:
            K (numpy.ndarray): The original kernel matrix (n x n).

        Returns:
            numpy.ndarray: The density-normalized kernel matrix (n x n).
        """

        # Compute the density estimates p by summing the values of each row
        p = np.sum(K, axis=1)
        
        # Ensure p is a column vector
        p = p.reshape(-1, 1)
        
        # Compute the outer product of the density estimates
        p_outer = np.sqrt(p @ p.T)
        
        # Compute the density-normalized kernel
        K_norm = K / p_outer
        
        return K_norm
    
    def hellinger_distance_matrix(self, matrix):
        """
        Compare each row to each other row in the matrix with the Hellinger algorithm, determining similarities between distributions.

        Parameters:
            matrix (numpy.ndarray): Matrix for the computation. Is expected to be the block.

        Returns:
            numpy.ndarray: Distance matrix.
        """

        #Create a single matrix by stacking the two blocks
        matrix = np.vstack([matrix[:self.len_A, :self.len_A], matrix[self.len_A:, self.len_A:]])

        #Reshape the maticies
        sqrt_matrix1 = np.sqrt(matrix[:, np.newaxis, :]) #NOTE: This will also take a good amount of memory
        sqrt_matrix2 = np.sqrt(matrix[np.newaxis, :, :])

        # Calculate the squared differences
        squared_diff = (sqrt_matrix1 - sqrt_matrix2) ** 2
        
        # Sum along the last axis to get the sum of squared differences
        sum_squared_diff = np.sum(squared_diff, axis=2)
        
        # Calculate the Hellinger distances
        distances = np.sqrt(sum_squared_diff) / np.sqrt(2)
        
        return distances

    def find_new_connections(self, pruned_connections = [], connection_limit = None, threshold = 0.2): 
        """
        A helper function that finds and returns a list of the closest connections and their associated weights after alignment.

        Parameters:
            pruned_connections (list): A list formatted like (n1, n2) where n1 is a point in Domain A, and n2 is a point in Domain B.
                                       The node connections in this list will not be considered for possible connections.
            connection_limit (int, optional): If set, the function will find no more than the specified number of connections.
            threshold (float): The threshold determines how similar a point has to be to another to be kept as a connection.

        Returns:
            numpy.ndarray: The possible connections.
        """

        #Keep Track of known-connections by creating a mask of everywhere we have a connection
        known_connections = self.similarity_matrix > 0

        if self.verbose > 1:
            print(f"Total number of Known_connections: {np.sum(known_connections)}")

        #This is made into an array to ensure the self.int_diff_dist is not changed
        array = np.array(self.int_diff_dist)

        #Set our Known-connections to inf values so they are not found and changed
        array[known_connections] = np.inf

        #Modify our array just to be the off-diagonal portion
        array = array[:self.len_A, self.len_A:]

        #Set the pruned_connections to be infinite as well
        array[pruned_connections] = np.inf

        #Set the connection_limit to be 1/3 of available connections if no limit was given
        if connection_limit == None:
            connection_limit = int((np.min(array.shape) - len(self.known_anchors)) / 3)

        # Flatten the array
        array_flat = array.flatten()

        # Sort the array so we can find the smallest values
        smallest_indices = np.argsort(array_flat)

        # Select the indices of the first smallest values equal to the connection limit
        smallest_indices = smallest_indices[:connection_limit]

        # Convert the flattened indices to tuple coordinates (row, column)
        coordinates = [np.unravel_index(index, array.shape) for index in smallest_indices]

        #Add in coordinate values as the third index in the tuple (row, column, value)
        coordinates = [(int(coordinate[0]), int(coordinate[1]), array[coordinate[0], coordinate[1]]) for coordinate in coordinates]

        #Select only coordinates whose values are less than the given threshold
        coordinates = np.array(list(takewhile(lambda x: x[2] < threshold, coordinates)))

        return coordinates

    """                                      <><><><><><><><><><><><><><><><><><><><>     
                                                    PRIMARY FUNCTIONS BELOW
                                             <><><><><><><><><><><><><><><><><><><><>                                                    """
    
    def merge_graphs(self): 
        """
        Creates a new graph (called graphAB) from graphs A and B using the known_anchors,
        adding an edge set with weight of 1 (as it is a similarity measure). It also connects
        its corresponding neighbors.

        Returns:
            numpy.ndarray: The kernel array of graphAB.
        """

        #convert Graphs to Igraphs
        graphA = self.graph_a.to_igraph()
        graphB = self.graph_b.to_igraph()

        #Merge the two graphs together in a disjoint way
        merged = graphA.disjoint_union(graphB)

        #For each anchor we want to find its neighbors (so we can connect those same edges to the anchor in the other domain).
        for anchor in self.known_anchors: 
            #Find the neighbors for each anchor
            neighborsA = tuple(set(graphA.neighbors(anchor[0], mode="out"))) #Anchor 0 applies to the graph A
            neighborsB = tuple(set(graphB.neighbors(anchor[1], mode="out"))) #Anchor 1 applies to the graph B

            #We add the edge weights first to a list so we can bulk add them later... NOTE: Since we are taking the weights from the kernal and not the graph object, it may be slightly different. It looks worse when we test the Projections, but the FOSCTTM score seems to be higher
            weights_to_add = self.kernalsB[neighborsB, np.repeat(anchor[1], len(neighborsB))]
            weights_to_add = np.append(weights_to_add, self.kernalsA[neighborsA, np.repeat(anchor[0], len(neighborsA))])

            #Bulk add the Edges
            edges_to_add = list(zip(np.full_like(neighborsB, anchor[0]), np.array(neighborsB) + self.len_A)) + list(zip(np.full_like(neighborsA, anchor[1]) + self.len_A, neighborsA)) #The self.len_A is to get it to correlate with the point in the domain B
            merged.add_edges(edges_to_add)
                
            #Add the weights
            merged.es[-len(weights_to_add):]["weight"] = weights_to_add

        #Now add the edges between anchors. We do this last so if we don't override an anchor in the previous step if multiple points in a domain correlate to a single point in the other domain.
        merged.add_edges(list(zip(self.known_anchors_adjusted[:, 0], self.known_anchors_adjusted[:, 1])))
        merged.es[-len(self.known_anchors_adjusted):]["weight"] = np.repeat(self.IDC, len(self.known_anchors_adjusted))

        #Convert back to graphtools
        merged_graphtools = graphtools.api.from_igraph(merged)

        return merged_graphtools.K.toarray()
    
    def get_diffusion(self, matrix, return_projection = True): 
        """
        Returns the powered diffusion operator from the given matrix.
        Also returns the projection matrix from domain A to B, and then the projection matrix from domain B to A.

        Parameters:
            matrix (numpy.ndarray): The input matrix.
            return_projection (bool, optional): Whether to return the projection matrices. Default is True.

        Returns:
            tuple: If return_projection is True, returns a tuple containing the diffused matrix, domainAB, and domainBA.
                Otherwise, returns only the diffused matrix.
        """

        #Find best T value if t is set to auto
        if self.t == -1 or type(self.t) != int:
            self.t = find_optimal_t(matrix) 

            #If we found a T
            if self.verbose > 1:
                print(f"Using optimal t value of {self.t}")
     
        # Row normalize the matrix
        normalized_matrix = self.row_normalize_matrix(matrix)

        #Apply the page rank algorithm
        if self.page_rank == "full":
            if self.verbose > 1:
                print("Applying Page Ranking against the full matrix")
            
            normalized_matrix = self.apply_page_rank(normalized_matrix)
            normalized_matrix = self.row_normalize_matrix(normalized_matrix)
        
        #Get off-Diagonal blocks and apply the Page Rank transformation
        elif self.page_rank == "off-diagonal":
            if self.verbose > 1:
                print("Applying Rage Ranking against the off-diagonal parts of the matrix")

            normalized_matrix[:self.len_A, self.len_A:] = self.apply_page_rank(normalized_matrix[:self.len_A, self.len_A:], alpha = 0.97) #Top right
            normalized_matrix[self.len_A:, :self.len_A] = self.apply_page_rank(normalized_matrix[self.len_A:, :self.len_A], alpha = 0.97) #Bottom left
            normalized_matrix = self.row_normalize_matrix(normalized_matrix)
            

        #Raise the normalized matrix to the t power
        diffusion_matrix = np.linalg.matrix_power(normalized_matrix, self.t)

        if return_projection:
            #Prepare the Projection Matricies by normalizing each domain by itself
            domainAB = diffusion_matrix[:self.len_A, self.len_A:]#Top Right
            domainBA = diffusion_matrix[self.len_A:, :self.len_A] #Bottom Left
            domainAB = self.row_normalize_matrix(domainAB)
            domainBA = self.row_normalize_matrix(domainBA)
        
        #Apply the aggregation function
        diffused = self.apply_aggregation(diffusion_matrix)
        
        if return_projection:
            return diffused, domainAB, domainBA
        else:
            return diffused

    def optimize_by_creating_connections(self, epochs = 100, threshold = "auto", connection_limit = "auto", hold_out_anchors = []):
        """
        In an iterative process, gets the potential anchors after alignment, and then recalculates the similarity matrix and diffusion operator.
        Tests this new alignment, and if it is better, keeps the alignment.

        Parameters:
            connection_limit (int or str, optional): If set, caps the max amount of anchors found. Best values to try: 1/10, 1/5, or 10x the length of the data, or None.
            threshold (float or str, optional): If auto, the algorithm will determine it. Cannot be higher than the median value of the dataset.
                                                Determines how similar a point has to be to another to be considered an anchor.
            hold_out_anchors (list, optional): Should be in the same format as known_anchors. Used to validate the new alignment. Performs best if these are unseen anchors.
            epochs (int, optional): The number of iterations the cycle will go through. Default is 3.

        Returns:
            bool: True if a new alignment was made, otherwise False.
        """

        #Create value to return
        added_connections = False

        #Show the original connections
        if self.verbose > 2:
            print("<><><><><> Beggining Tests. Original Connections show below <><><><><>")
            plt.imshow(self.similarity_matrix)
            plt.show()

        
        #Set pruned_connections to equal hold_out_anchhor connections if they exist, empty otherwise
        if len(hold_out_anchors) > 0:

            #First add the hold_out_anchor connections
            pruned_connections = list(hold_out_anchors)

            #Create empty lists that will hold the anchor's neighbors (because these are also known connections)
            hold_neighbors_A = []
            hold_neighbors_B = []

            #Add in the the connections of each neighbor to each anchor
            for anchor_pair in hold_out_anchors:

                #Cache the data
                hold_neighbors_A += [(neighbor, anchor_pair[1]) for neighbor in set(self.graph_a.to_igraph().neighbors(anchor_pair[0], mode="out"))]
                hold_neighbors_B += [(anchor_pair[0], neighbor) for neighbor in set(self.graph_b.to_igraph().neighbors(anchor_pair[1], mode="out"))]

            #Add the connections
            pruned_connections += hold_neighbors_A
            pruned_connections += hold_neighbors_B
            
            #Convert to Numpy array for advanced indexing (for later use)
            hold_neighbors_A = np.array(hold_neighbors_A)
            hold_neighbors_B = np.array(hold_neighbors_B)
            
        else:
            pruned_connections = []

        if threshold == "auto":
            #Set the threshold to be the 10% limit of the connections
            threshold = np.sort(self.int_diff_dist.flatten())[:int(len(self.int_diff_dist.flatten()) * .1)][-1]

        if connection_limit == "auto":
            #Set the connection limit to be 10x the shape (while not always the best value, its consistently good and much faster due to the need of using less epochs)
            connection_limit = 10 * self.len_A

        #Get the current score of the alignment, by calculating the FOSCTTM scores that correlate with the hold_out_anchors
        current_score = np.mean([self.partial_FOSCTTM(self.int_diff_dist[self.len_A:, :self.len_A], hold_out_anchors), self.partial_FOSCTTM(self.int_diff_dist[:self.len_A, self.len_A:], hold_out_anchors)])

        #Find the Max value for new connections to be set to
        max_weight = np.median(self.similarity_matrix[self.similarity_matrix != 0])

        #Make sure we aren't finding values greater than the max_weight
        if threshold > max_weight:
            max_weight = threshold + 0.01
        
        if self.verbose > 0:
            print(f"Edges wont be set with similarity measure above: {max_weight}")

        #-----------------------------------------------------------------      Rebuild Class for each epoch        -----------------------------------------------------------------    
        for epoch in range(0, epochs):
            
            if self.verbose > 1:
                print(f"<><><><><><><><><><><><>    Starting Epoch {epoch}    <><><><><><><><><><><><><>")

            #Find predicted anchors
            new_connections = self.find_new_connections(pruned_connections, threshold = threshold, connection_limit = connection_limit)

            #If no new connections are found, quit the process
            if len(new_connections) < 1:
                if self.verbose > 1:
                    print("No new_connections. Exiting process")

                #Add in the known anchors and reset the known_anchors, similarity_matrix, and diffusion matrix
                if len(hold_out_anchors) > 0:

                    #Cached info 
                    adjusted_hold_neighbors_B = hold_neighbors_B + self.len_A

                    #Set the connections values to the top right block
                    self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1] + self.len_A] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]
                    self.similarity_matrix[hold_neighbors_A[:, 0] + self.len_A, hold_neighbors_A[:, 1]] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]

                    #Set the connection values to the bottom left block
                    self.similarity_matrix[hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]
                    self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]

                    #Set the anchors
                    self.similarity_matrix[hold_out_anchors[:, 0], hold_out_anchors[:, 1] + self.len_A] = self.IDC
                    self.similarity_matrix[hold_out_anchors[:, 0] + self.len_A, hold_out_anchors[:, 1]] = self.IDC

                    #Reset the Diffusion Matrix
                    self.int_diff_dist, self.projectionAB, self.projectionBA = self.get_diffusion(self.similarity_matrix)

                    #Add in the hold out anchors to the known_anchors
                    self.known_anchors = np.concatenate([self.known_anchors, hold_out_anchors])

                    #readjust known_anchors to correspond to off diagonal matricies
                    self.known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

                #Return True if we had found a new alignment, otherwise false
                return added_connections

            #Continue to show connections
            if self.verbose > 1:
                print(f"New connections found: {len(new_connections)}")

            #Copy Similarity matrix
            new_similarity_matrix = np.array(self.similarity_matrix) #We do this redudant conversion to an array to ensure we aren't copying over a reference

            #Add the new connections
            new_similarity_matrix[new_connections[:, 0].astype(int), (new_connections[:, 1] + self.len_A).astype(int)] = max_weight - new_connections[:, 2] #The max_weight minus is supposed to help go from distance to similarities
            new_similarity_matrix[(new_connections[:, 0] + self.len_A).astype(int) , new_connections[:, 1].astype(int)] = max_weight - new_connections[:, 2] #This is so we get the connections in the other off-diagonal block

            #Show the new connections
            if self.verbose > 2:
                plt.imshow(new_similarity_matrix)
                plt.show()

            #Get new Diffusion Matrix
            new_int_diff_dist = self.get_diffusion(new_similarity_matrix, return_projection = False)

            #Get the new alignment score
            new_score = np.mean([self.partial_FOSCTTM(new_int_diff_dist[self.len_A:, :self.len_A], hold_out_anchors), self.partial_FOSCTTM(new_int_diff_dist[:self.len_A, self.len_A:], hold_out_anchors)])

            #See if the extra connections helped
            if new_score < current_score or len(hold_out_anchors) < 1:

                if self.verbose > 1:
                    print(f"The new connections improved the alignment by {current_score - new_score}\n-----------     Keeping the new alignment. Continuing...    -----------\n")

                #Reset all the class variables. We don't worry about the calculating the projection matricies until the last epoch.
                self.similarity_matrix = new_similarity_matrix
                self.int_diff_dist = new_int_diff_dist

                #Reset the score
                current_score = new_score

                #Change Added connections to True
                added_connections = True

            else:
                if self.verbose > 1:
                    print(f"The new connections worsened the alignment by {new_score - current_score}\n-----------     Pruning the new connections. Continuing...    -----------\n")

                #Add the added connections to the the pruned_connections
                if len(pruned_connections) < 1:
                    pruned_connections = new_connections[:, :2].astype(int)
                else:
                    pruned_connections = np.concatenate([pruned_connections, new_connections[:, :2]]).astype(int)

        #On the final epoch, we can evaluate with the hold_out_anchors and then assign them as anchors. Also we need to ensure we calculate the projection matricies
        if epoch == epochs - 1:

            #Add in hold_out_anchors if applicable
            if len(hold_out_anchors) > 0:
                #Cached info 
                adjusted_hold_neighbors_B = hold_neighbors_B + self.len_A

                #Set the connections values to the top right block
                self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1] + self.len_A] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]
                self.similarity_matrix[hold_neighbors_A[:, 0] + self.len_A, hold_neighbors_A[:, 1]] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]

                #Set the connections values to the bottom left block
                self.similarity_matrix[hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]
                self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]

                #Set the anchors
                self.similarity_matrix[hold_out_anchors[:, 0], hold_out_anchors[:, 1] + self.len_A] = self.IDC
                self.similarity_matrix[hold_out_anchors[:, 0] + self.len_A, hold_out_anchors[:, 1]] = self.IDC

            #Recalculate diffusion matrix
            self.int_diff_dist, self.projectionAB, self.projectionBA = self.get_diffusion(self.similarity_matrix)

            #Show the final connections
            if self.verbose > 1:
                print("Added Hold Out Anchor Conections")
                plt.imshow(self.similarity_matrix)
                plt.show()
            
        #Process Finished
        if self.verbose > 0:
            print("<><><><><><><><><><<><><><><<> Epochs Finished <><><><><><><><><><><><><><><><><>")

        #Add in the hold out anchors to the known_anchors
        self.known_anchors = np.concatenate([self.known_anchors, hold_out_anchors])

        #readjust known_anchors to correspond to off diagonal matricies
        self.known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

        return added_connections

    """                                      <><><><><><><><><><><><><><><><><><><><>     
                                                    PRIMARY FUNCTIONS BELOW
                                             <><><><><><><><><><><><><><><><><><><><>                                                    """
    def project_feature(self, predict_with = "A"):
        """
        Project the feature values from one domain to the other using the projection matrices.

        NOTE: This function is in experimentation.

        Parameters:
            predict_with (str): Specifies which graph data to use. 'A' for graph A and 'B' for graph B.

        Returns:
            numpy.ndarray: The predicted features in an array.
        """

        if predict_with == "A":
            known_features = self.dataA
            projection_matrix = self.projectionBA #Bottom Left
        elif predict_with == "B":
            known_features = self.dataB
            projection_matrix = self.projectionAB #Top Right
        else:
            print("Please specify which features you want to predict. Graph 'A' or Graph 'B'")
            return None
        
        
        predicted_features = (projection_matrix[:, :, np.newaxis] * known_features[np.newaxis, :, :]).sum(axis = 1)
        
        return predicted_features
    
    def get_merged_data_set(self):
        """
        Adds the predicted features to the datasets with the missing features.
        
        Returns:
            numpy.ndarray: A combined dataset that includes the predicted features.
        """
        #Add the predicted features to each data set
        full_data_A = np.hstack([self.predict_feature(predict = 'A'), self.dataB])
        full_data_B = np.hstack([self.dataA, self.predict_feature(predict = 'B')])

        #Combine the datasets
        completeData = np.vstack([full_data_A, full_data_B])

        return completeData

    """                                     <><><><><><><><><><><><><><><><><><><><><>     
                                            VISUALIZATION AND SCORING FUNCTIONS BELOW
                                            <><><><><><><><><><><><><><><><><><><><><>                                                    """
    def plot_heat_maps(self):
        """
        Plots and shows the heat maps for the similarity matrix, powered diffusion operator,
        and projection matrix.
        """
        fig, axes = plt.subplots(1, 3, figsize = (13, 9))

        #Similarity matrix
        axes[0].imshow(self.similarity_matrix)
        axes[0].set_title("Similarity Matrix")

        #Projection AB
        axes[2].imshow(self.projectionAB)
        axes[2].set_title("Projection AB")

        #Diffusion matrix
        axes[1].imshow(self.int_diff_dist)
        axes[1].set_title("Integrated Diffusion Distance Matrix")

        plt.show()

    def plot_emb(self, labels = None, n_comp = 2, show_lines = True, show_anchors = True, show_pred = False, show_legend = True, **kwargs): 
        """
        A useful visualization function to view the embedding.

        Parameters:
            labels (list, optional): A flattened list of the labels for points in domain A and then domain B.
                                    If set to None, the cross embedding cannot be calculated, and all points will be colored the same.
            n_comp (int, optional): The number of components or dimensions for the MDS function.
            show_lines (bool, optional): If set to True, plots lines connecting the points that correlate to the points in the other domain.
                                        Assumes a 1 to 1 correspondence.
            show_anchors (bool, optional): If set to True, plots a black square on each point that is an anchor.
            show_pred (bool, optional): If set to True, shows labels in Domain B as muted, and then plots a second graph with Domain B
                                        with its predicted labels.
            show_legend (bool, optional): If set to True, displays the legend.
            **kwargs: Additional keyword arguments for sns.scatterplot function.

        Returns:
            None
        """

        #Add show_legend to kwargs
        if "legend" not in kwargs.keys():
           kwargs["legend"] = show_legend

        FOSCTTM_score, CE_score, RF_score = self.get_scores(labels, n_comp = n_comp)

        print(f"Cross Embedding score: {CE_score}")
        print(f"Fraction of Samples Closest to thier Match: {FOSCTTM_score}")
        print(f"RF score trained on full Embedding: {RF_score}")


        if type(labels) == type(None):
            #Set all labels to be the same
            labels = np.ones(shape = (len(self.emb)))

        #Set the styles to show if a point comes from the first domain or the second domain
        styles = ['Domain A' if i < self.len_A else 'Domain B' for i in range(len(self.emb[:]))]

        #Create the figure
        plt.figure(figsize=(14, 8))

        #If show_pred is chosen, we want to show labels in Domain B as muted
        if show_pred:
            ax = sns.scatterplot(x = self.emb[self.len_A:, 0], y = self.emb[self.len_A:, 1], color = "grey", s=120, marker= "o", **kwargs)
            ax = sns.scatterplot(x = self.emb[:self.len_A, 0], y = self.emb[:self.len_A, 1], hue = Categorical(labels[:self.len_A]), s=120, marker= "^", **kwargs)
        else:
            #Now plot the points with correct lables
            ax = sns.scatterplot(x = self.emb[:, 0], y = self.emb[:, 1], style = styles, hue = Categorical(labels), s=120, markers= {"Domain A": "^", "Domain B" : "o"}, **kwargs)

        #Set the title and plot Legend
        ax.set_title("MASH", fontsize = 25)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        #Plot Legend
        if show_legend:
            plt.legend()

        #To plot line connections
        if show_lines:
            
            #Since this assumes 1 to 1 correpsondence, we must chech that the domains sizes are the same
            if self.len_A == self.len_B:
              for i in range(self.len_B):
                  ax.plot([self.emb[0 + i, 0], self.emb[self.len_A + i, 0]], [self.emb[0 + i, 1], self.emb[self.len_A + i, 1]], alpha = 0.65, color = 'lightgrey') #alpha = .5
            else:
               raise AssertionError("To show the lines, domain A and domain B must be the same size.")
             
        #Put black dots on the Anchors
        if show_anchors:
            
            #For each anchor set, plot lines between them
            for i in self.known_anchors_adjusted:
              ax.plot([self.emb[i[0], 0], self.emb[i[1], 0]], [self.emb[i[0], 1], self.emb[i[1], 1]], color = 'grey')
            
            #Create a new style guide so every other point is a triangle or circle
            styles2 = ['Domain A' if i % 2 == 0 else 'Domain B' for i in range(len(self.known_anchors)*2)]

            #Plot the black triangles or circles on the correct points
            sns.scatterplot(x = np.array(self.emb[self.known_anchors_adjusted, 0]).flatten(), y = np.array(self.emb[self.known_anchors_adjusted, 1]).flatten(), style = styles2, linewidth = 2, markers= {"Domain A": "x", "Domain B" : "+"}, s = 45, color = "black", **kwargs)
        
        #Show plot
        plt.show()

        #Show the predicted points
        if show_pred and type(labels) != type(None):

            #Instantial model, fit on domain A, and predict for domain B
            knn_model = KNeighborsClassifier(n_neighbors=4)
            knn_model.fit(self.emb[:self.len_A, :], labels[:self.len_A])
            second_pred = knn_model.predict(self.emb[self.len_A:, :])

            #Create the figure
            plt.figure(figsize=(14, 8))

            #Now plot the points
            ax = sns.scatterplot(x = self.emb[:, 0], y = self.emb[:, 1], style = styles, hue = Categorical(np.concatenate([labels[:self.len_A], second_pred])), s=120, markers= {"Domain A": "^", "Domain B" : "o"}, **kwargs)

            #Set the title
            ax.set_title("Predicted Labels",  fontsize = 25)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.show()

    def plot_t_grid(self, rate = 5):
        """Plots the powered diffusion operator many times each with a different t value. Also plots
        the associated projection matrix. 
        
        Arguments:
            :rate: the value by which to increment t for each iteration.
            
        It has no return.
        """

        #Store the original t value
        t_str = str(self.t)
        
        #Create the figure
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(13, 17))

        #Create an empty list to store the FOSCTTM scores
        F_scores = np.array([])

        for i in range(1, 6):

            # Perform the diffusion operation
            self.t = i * rate
            diffused_array = self.get_diffusion(self.similarity_matrix, return_projection=False)

            #Calculate FOSCTTM score
            F_score = self.FOSCTTM(diffused_array[self.len_A:, :self.len_A])
            
            # Plotting the diffused array
            ax = axes[i-1]
            ax.imshow(diffused_array)
            ax.set_title(f'T value {self.t}, FOSCTTM {(F_score):.4g}', fontsize = 10)
            ax.axis('off')

            
        #Show the plot
        plt.tight_layout()
        plt.show()

        #Restore t value
        self.t = int(t_str)

    def get_scores(self, labels = None, n_comp = 2):
        """
        Returns the FOSCTTM score and Cross_embedding Score. If labels are not provided, the Cross Embedding will be returned as None.

        Parameters
        ----------
        labels : array-like, optional
            The labels for the dataset. If labels are not provided, just the FOSCTTM score will be returned.
        n_comp : int, optional
            The number of components for the MDS.

        Returns
        -------
        tuple
            FOSCTTM score and Cross_embedding Score.
        """

        #Check to see if we already have created our embedding, else create the embedding.
        if type(self.emb) == type(None):
            self.print_time()
            #Convert to a MDS
            mds = MDS(metric=True, dissimilarity = 'precomputed', n_components= n_comp, random_state = self.kwargs["random_state"])
            self.emb = mds.fit_transform(self.int_diff_dist)
            self.print_time("Time it took to calculate the embedding: ")

        #Check to make sure we have labels
        if type(labels)!= type(None):
            #Seperate the labels into their respective domains
            first_labels = np.array(labels[:self.len_A])
            second_labels = np.array(labels[self.len_A:])

            #Calculate Cross Embedding Score
            try: 
                CE_score = self.cross_embedding_knn(self.emb, (first_labels, second_labels), knn_args = {'n_neighbors': 5})
            except:
                CE_score = None
        else:
            CE_score = None

        #RF Gap trained on full embedding
        if type(labels[0]) != int:
            rf_class = RFGAP(prediction_type="regression", y=labels, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=True, oob_score = True)
        else:
            rf_class = RFGAP(prediction_type="classification", y=labels, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=True, oob_score = True)

        #Fit it for Data A and get proximities
        rf_class.fit(self.emb, y = labels)

        #Calculate FOSCTTM score
        try:    
            FOSCTTM_score = self.FOSCTTM(self.int_diff_dist[self.len_A:, :self.len_A])
        except: #This will run if the domains are different shapes
            FOSCTTM_score = None

        return FOSCTTM_score, CE_score, rf_class.oob_score_

    """                                     <><><><><><><><><><><><><><><><><><><><>     
                                                   EVALUATION FUNCTIONS BELOW
                                            <><><><><><><><><><><><><><><><><><><><>                                                    """
    
    def FOSCTTM(self, off_diagonal): 
        """
        FOSCTTM stands for average Fraction of Samples Closer Than the True Match.

        Lower scores indicate better alignment, as similar or corresponding points are mapped closer 
        to each other through the alignment process. If a method perfectly aligns all corresponding 
        points, the average FOSCTTM score would be 0.

        Parameters
        ----------
        off_diagonal : array-like
            Should be either off-diagonal portion (that represents mapping from one domain to the other)
            of the block matrix.

        Returns
        -------
        float
            The average FOSCTTM score.
        """

        n1, n2 = np.shape(off_diagonal)
        if n1 != n2:
            raise AssertionError('FOSCTTM only works with a one-to-one correspondence. ')

        dists = off_diagonal

        nn = NearestNeighbors(n_neighbors = n1, metric = 'precomputed')
        nn.fit(dists)

        _, kneighbors = nn.kneighbors(dists)

        return np.mean([np.where(kneighbors[i, :] == i)[0] / n1 for i in range(n1)])
    
    def partial_FOSCTTM(self, off_diagonal, anchors):
        """
        Follows the same format as FOSCTTM.

        This calculates only a subset of points. It is intended to be used with hold-out anchors to help 
        us gauge whether new connections yield in a better alignment.

        Parameters
        ----------
        off_diagonal : array-like
            Should be either off-diagonal portion (that represents mapping from one domain to the other)
            of the block matrix.
        anchors : list of tuples
            List of anchor points to calculate the partial FOSCTTM score.

        Returns
        -------
        float
            The average partial FOSCTTM score.
        """

        n1, n2 = np.shape(off_diagonal)
        if n1 != n2:
            raise AssertionError('FOSCTTM only works with a one-to-one correspondence. ')

        dists = off_diagonal

        nn = NearestNeighbors(n_neighbors = n1, metric = 'precomputed')
        nn.fit(dists)

        _, kneighbors = nn.kneighbors(dists)

        return np.mean([np.where(kneighbors[i[0], :] == i[1])[0] / n1 for i in anchors])
    
    def cross_embedding_knn(self, embedding, Labels, knn_args = {'n_neighbors': 4}):
      """
      Returns the classification or regression score by training on one domain and predicting on the other.

      Parameters
      ----------
      embedding : array-like
          The embedding of the data.
      Labels : tuple of array-like
          The labels for the two domains.
      knn_args : dict, optional
          Arguments for the k-nearest neighbors classifier.

      Returns
      -------
      float
          The cross_embedding_knn score.
      """

      (labels1, labels2) = Labels

      n1 = len(labels1)

      # Determine if the task is classification or regression
      if np.issubdtype(labels1.dtype, np.integer):
          # Classification
          knn = KNeighborsClassifier(**knn_args)

          if self.verbose > 2:
             print("Calculating the classification Score.")
      else:
          # Regression
          knn = KNeighborsRegressor(weights = "distance", **knn_args)

          if self.verbose > 2:
             print("Calculating the R squared score.")

      #Fit and score predicting from domain A to domain B
      knn.fit(embedding[:n1, :], labels1)
      score1 =  knn.score(embedding[n1:, :], labels2)

      #Fit and score predicting from domain B to domain A, and then return the average value
      knn.fit(embedding[n1:, :], labels2)
      return np.mean([score1, knn.score(embedding[:n1, :], labels1)])

