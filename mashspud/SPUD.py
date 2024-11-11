"""
Shortest Path to Union Domains (SPUD)

A class that learns the inter-geodesic distances between manifolds. Similar to MASH. 
"""

#Install the libraries
from scipy.spatial.distance import pdist, squareform, _METRICS
import graphtools
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from .triangular_helper import *
from .RF_GAP import RFGAP

#Additional Libraries to support plotting and verbose levels
from time import time
import seaborn as sns
from pandas import Categorical

class SPUD:
  def __init__(self, distance_measure_A = "euclidean", distance_measure_B = "euclidean", knn = 5,
               OD_method = "default", agg_method = "log", IDC = 1, #TODO: See if its possible to get rid of either OD_method or similarity Measure
               overide_method = "none", #Maybe name this method and then have Jaccard, NAMA, and SPUD be seperate choosable methods
               float_precision = np.float32, verbose = 0, **kwargs):
        """
        Initializes the class object.

        Parameters
        ----------
        distance_measure_A : str or callable, optional
            Distance measure for domain A. Can be a function, "precomputed", or a SciKit-learn metric string.
        distance_measure_B : str or callable, optional
            Distance measure for domain B. Can be a function, "precomputed", or a SciKit-learn metric string.
        knn : int or str, optional
            Number of nearest neighbors for graph construction. If "connect", ensures graph connection.
        OD_method : str, optional
            Off-diagonal method. Options are "absolute_distance", "mean", or "default".
        agg_method : str or float, optional
            Method to adjust off-diagonal blocks in alignment. Options are 'sqrt', 'log', any float, or 'None'.
        IDC : float, optional
            Inter-domain correspondence value for anchor points between domains.
        overide_method : str, optional
            Method to override default. Options are "none", "NAMA", "similarities", or "Jaccard".
        float_precision : dtype, optional
            Precision of floating-point numbers.
        verbose : int or float, optional
            Level of verbosity for output. Expected values are 0,1,2,3,4.
        **kwargs : dict, optional
            Additional keyword arguments for the graphtools.Graph function.
        """

        #Set the values
        self.distance_measure_A = distance_measure_A
        self.distance_measure_B = distance_measure_B
        self.verbose = verbose
        self.knn = knn
        self.agg_method = agg_method
        self.kwargs = kwargs
        self.IDC = IDC
        self.OD_method = OD_method.lower()
        self.overide_method = overide_method.lower()
        self.float_precision = float_precision

        #Set self.emb to be None
        self.emb = None

        #Ensure there is a random state
        if "random_state" not in self.kwargs.keys():
            self.kwargs["random_state"] = 42

        #Set precomputed to be none if override method is set
        if self.overide_method != "none":
          if "precomputed" not in self.kwargs.keys():
              self.kwargs["precomputed"] = "distance"

        #Adjust the values to work together
        if self.overide_method != "none" and self.OD_method == "default":
          if self.verbose > 0:
             print(f"Setting the off-diagonal method (OD_method) to 'absolute_distance' to be compatible with {self.overide_method} method.")
          
          self.OD_method = "absolute_distance"

  def fit(self, dataA, dataB, known_anchors):
        """
        Computes the manifold alignment using shortest path distances.

        Parameters
        ----------
        dataA : array-like
            The data for domain A.
        dataB : array-like
            The data for domain B.
        known_anchors : list of tuples
            Points in domain A that correlate to domain B. Should be formatted as (n, 2), where n is the number of points that correspond.
            For any nth position, the 0th place represents the point in domain A and the 1st position represents the point in domain B.
            Example: [[1,1], [4,3], [7,6]].
        """

        #Cache these values for fast lookup
        self.len_A = len(dataA) 
        self.len_B = len(dataB)

        #Save the known Anchors
        self.known_anchors = np.array(known_anchors)

        #For each domain, calculate the distances within their own domain
        self.print_time()
        self.distsA = self.get_SGDM(dataA, self.distance_measure_A)
        self.distsB = self.get_SGDM(dataB, self.distance_measure_B)
        self.print_time("Time it took to compute SGDM:  ")

        #Change known_anchors to correspond to off diagonal matricies. 
        self.known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

        #Check to make sure the anchors are given correctly
        if np.max(self.known_anchors[:, 0]) > self.len_A or  np.max(self.known_anchors[:, 1]) > self.len_B:
           raise RuntimeWarning("Warning: Check you known anchors. Anchors given exceed vertices in data.")

        #If these parameters are true, we can skip the graph creation
        if self.overide_method == "nama":
           if self.verbose > 0:
              print("Skipping graph creating. Performing nearest anchor manifold alignment (NAMA) instead of SPUD.")

        else:
          #Create Igraphs and kernals from the input.
          self.print_time()
          self.graphA = graphtools.Graph(reconstruct_symmetric(self.distsA), knn = self.knn, knn_max= self.knn, **self.kwargs)
          self.graphB = graphtools.Graph(reconstruct_symmetric(self.distsB), knn = self.knn, knn_max= self.knn, **self.kwargs)

          if "precomputed" not in self.kwargs.keys():
            self.kernalsA = get_triangular(self.graphA.K.toarray())
            self.kernalsB = get_triangular(self.graphB.K.toarray())
          else:
            self.kernalsA = get_triangular(self.graphA.K)
            self.kernalsB = get_triangular(self.graphB.K)

          self.graphA = self.graphA.to_igraph()
          self.graphB = self.graphB.to_igraph()
          self.print_time("Time it took to execute graphtools.Graph functions:  ")

        #Merge the graphs
        if self.OD_method == "default":

          if self.verbose > 0 and self.len_A > 1500:
             raise ResourceWarning("Computing off-diagonal blocks will be exspensive. Consider setting OD_method to 'mean' or 'absolute_distance' for faster computation time.")
             
          self.print_time()
          self.graphAB = self.merge_graphs()
          self.print_time("Time it took to execute merge_graphs function:  ")

        #Get the distances
        self.print_time()
        self.block = self.get_block()
        self.print_time("Time it took to execute get_block function:  ")

        if self.verbose > 0:
           print("<><><><><><><><><><><><><>  Processed Finished  <><><><><><><><><><><><><>")

  def get_distsA(self):
     """Returns the reconstructed distance matrix for domain A"""
     return reconstruct_symmetric(self.distsA)
  
  def get_distsB(self):
     """Returns the reconstructed distance matrix for domain B"""
     return reconstruct_symmetric(self.distsB)
  
  def get_kernalsA(self):
     """Returns the reconstructed distance matrix for kernal A"""
     return reconstruct_symmetric(self.kernalsA)
  
  def get_kernalsB(self):
     """Returns the reconstructed distance matrix for kernal B"""
     return reconstruct_symmetric(self.kernalsB)
  
  """                                       <><><><><><><><><><><><><><><><><><><><><>     
                                                      HELPER FUNCTIONS BELOW
                                            <><><><><><><><><><><><><><><><><><><><><>                                                    """
  def print_time(self, print_statement =  ""):
    """
    Times the algorithms and prints how long it has been since the function was last called.

    Parameters:
        print_statement (str): A statement to print before the timing information.
    """

    #Only do this if the verbose is higher than 4
    if self.verbose > 3:

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
        time_string = str(round(end_time - self.start_time, 2))

        #Reset the start time
        self.start_time = None

        print(print_statement + time_string)

  def normalize_0_to_1(self, value):
    """
    Normalizes the value to be between 0 and 1 and resets infinite values.
    """

    #Scale it and check to ensure no devision by 0
    if np.max(value[~np.isinf(value)]) != 0:
      value = (value - value.min()) / (value[~np.isinf(value)].max() - value.min()) 

    #Reset inf values
    value[np.isinf(value)] = 1

    return value

  def get_SGDM(self, data, distance_measure):
    """
    SGDM - Same Graph Distance Matrix.
    Returns the normalized distances within each domain.

    Parameters
    ----------
    data : array-like
        The data for which the distance matrix is to be computed.
    distance_measure : str or callable
        The distance measure to use. Can be a function, "precomputed", or a SciKit-learn metric string.

    Returns
    -------
    array-like
        The normalized distance matrix.
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
           print("Warning. NaN's dectected. Calculating distances by ignoring NaN positions, and normalizing. May take longer.")

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
      raise RuntimeError(f"Did not understand {distance_measure}. Please provide a function, or use strings 'precomputed', or provided by sk-learn.")

    #Normalize it and return the data
    return get_triangular(self.normalize_0_to_1(dists))

  """                                       <><><><><><><><><><><><><><><><><><><><><>     
                                                   EVALUATION FUNCTIONS BELOW
                                            <><><><><><><><><><><><><><><><><><><><><>                                                    """
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

  """                                       <><><><><><><><><><><><><><><><><><><><><>     
                                                      PRIMARY FUNCTIONS BELOW
                                            <><><><><><><><><><><><><><><><><><><><><>                                                    """
  def merge_graphs(self):
        """
        Creates a new graph (called graphAB) from graphs A and B using the known_anchors,
        adding an edge set with weight of 1 (as it is a similarity measure).

        Returns:
            numpy.ndarray: The kernel array of graphAB.
        """

        #Merge the two graphs together
        merged = self.graphA.disjoint_union(self.graphB)

        #Now add the edges between anchors and set their  weight to 1
        merged.add_edges(list(zip(self.known_anchors_adjusted[:, 0], self.known_anchors_adjusted[:, 1])))
        merged.es[-len(self.known_anchors_adjusted):]["weight"] = np.repeat(self.IDC, len(self.known_anchors_adjusted))

        #Return the Igraph object
        return merged
    
  def get_off_diagonal_distances(self):
    """
    Calculates the off-diagonal by finding the closest anchors to each other.

    Parameters
    ----------
    None

    Returns
    -------
    array-like
        The off-diagonal distance matrix.
    """

    #The algorithm uses the kernals for speed and efficiency (so we don't waste time calculating similarities twice.)
    if self.verbose > 2:
       print(f"Preforming {self.OD_method} calculations.\n")

    #Set the matrix domains to equal the jaccard similarity
    if self.overide_method == "jaccard":
       matrixA = 1 - get_triangular(np.array(self.graphA.similarity_jaccard(pairs=None), dtype = self.float_precision))
       matrixB = 1 -  get_triangular(np.array(self.graphB.similarity_jaccard(pairs=None), dtype = self.float_precision))
    
    #Set the matrix domains to equal the distances via shortest paths
    elif self.overide_method == "distances" or self.overide_method == "none":
       matrixA = get_triangular(self.normalize_0_to_1(np.array(self.graphA.distances(weights = "weight"), dtype = self.float_precision)))
       matrixB = get_triangular(self.normalize_0_to_1(np.array(self.graphB.distances(weights = "weight"), dtype = self.float_precision)))
  
    #Just use the pure distance measure
    elif self.overide_method == "nama":
      matrixA = self.distsA
      matrixB = self.distsB

    elif self.overide_method == "similarities":
      matrixA = 1 - self.kernalsA
      matrixB = 1 - self.kernalsB

      #Change the interclass distances to be the kernals
      self.distsA = matrixA
      self.distsB = matrixB

    else: 
      raise RuntimeError(f"Did not understand the overide method: {self.overide_method}. Please use 'distances' (or 'none'), 'nama', 'similarities', or 'jaccard'")


    """Preform the absolute_distance and mean methods"""

    #Take the mean of each one first, then select.
    anchor_dists_A = get_triangular_mean(*index_triangular(matrixA, columns = self.known_anchors[:, 0], return_indices=True)) 
    anchor_dists_B = get_triangular_mean(*index_triangular(matrixB, columns = self.known_anchors[:, 1], return_indices=True))

    #Strecth A and B to be the correct sizes, and so each value matches up with each other value.
    anchor_dists_A = np.repeat(anchor_dists_A.astype(self.float_precision), repeats= self.len_B)
    anchor_dists_B = np.tile(anchor_dists_B.astype(self.float_precision), self.len_A)

    if self.OD_method == "absolute_distance":

      #Convert it to the square matrix. NOTE: We chose not to convert this back into a triangular. If memory is a concern, we recommend you add "reconstruct triagular" method here and adjust the code accordingly.
      off_diagonal = np.reshape(np.abs(anchor_dists_A - anchor_dists_B), newshape=(self.len_A, self.len_B))

    elif self.OD_method == "mean":
      
      #Convert it to the square matrix. NOTE: We chose not to convert this back into a triangular. If memory is a concern, we recommend you add "reconstruct triagular" method here and adjust the code accordingly.
      off_diagonal = np.reshape((anchor_dists_A + anchor_dists_B)/2, newshape=(self.len_A, self.len_B))

    else:
       raise RuntimeError("Did not understand your input for OD_method (Off-Diagonal method). Please use 'mean', 'absolute_distance', or 'default'.")
             
    return off_diagonal

  def get_block(self):
    """
    Returns a transformed and normalized block.

    Parameters
    ----------
    graph : Graph
        A graph that has merged together domains A and B.

    Returns
    -------
    array-like
        The transformed and normalized block.
    """

    #Find the off_diagonal block depending on our method
    if self.OD_method != "default":
       off_diagonal = self.get_off_diagonal_distances()
    else:
      #Get the vertices to find the distances between graphs. This helps when len_A != len_B
      verticesA = np.array(range(self.len_A))
      verticesB = np.array(range(self.len_B)) + self.len_A

      #Get the off-diagonal block by using the distance method. This returns a distnace matrix.
      #TODO: Think about how we neeed to reconstruct this
      off_diagonal = self.normalize_0_to_1(np.array(self.graphAB.distances(source = verticesA, target = verticesB, weights = "weight"))) # We could break this apart as another function to calculate the abs value in another way. This would reduce time complexity, though likely not be as accurate. 

    #Apply agg_method modifications
    if type(self.agg_method) == float:
      off_diagonal *= self.agg_method

    elif self.agg_method == "sqrt":
      off_diagonal = np.sqrt(off_diagonal + 1) #We have found that adding one helps

      #And so the distances are correct, we lower it so the scale is closer to 0 to 1
      off_diagonal = off_diagonal - off_diagonal.min()

    #If it is log, we check to to see if the domains match. If they do, we just apply the algorithm to the off-diagonal, which yeilds better results
    elif self.agg_method == "log" and self.len_A == self.len_B:
        #Apply the negative log, pdist, and squareform
        off_diagonal = self.normalize_0_to_1((squareform(pdist((-np.log(1+off_diagonal))))))

    elif self.agg_method == "normalize":
       off_diagonal = self.normalize_0_to_1(off_diagonal)

    #Recreate the block matrix --> This may be faster?
    #off_diagonal = reconstruct_symmetric(off_diagonal)

    #Create the block
    block = np.block([[reconstruct_symmetric(self.distsA), off_diagonal],
                        [off_diagonal.T, reconstruct_symmetric(self.distsB)]])
    
    #If the agg_method is log, and the domain shapes don't match, we have to apply the process to the block. 
    if self.agg_method == "log" and self.len_A != self.len_B:
      if self.verbose > 0:
         print("Domain sizes dont macth. Will apply the 'log' aggregation method against the whole block rather just the off_diagonal.")

      #Apply the negative log, pdist, and squareform
      block = self.normalize_0_to_1((squareform(pdist((-np.log(1+block))))))

    return block

  """                                       <><><><><><><><><><><><><><><><><><><><><>     
                                            VISUALIZATION AND SCORING FUNCTIONS BELOW
                                            <><><><><><><><><><><><><><><><><><><><><>                                                    """
  def plot_graphs(self):
    """
    Using the Igraph plot function to plot graphs A, B, and AB. 
    """
    #Create figure
    fig, axes = plt.subplots(1, 3, figsize = (19, 12))

    #Plot each of the plots
    ig.plot(self.graphA, vertex_color=['green'], target=axes[0], vertex_label= list(range(self.len_A)))
    ig.plot(self.graphB, vertex_color=['cyan'], target=axes[1], vertex_label= list(range(self.len_B)))
    ig.plot(self.graphAB, vertex_color=['orange'], target=axes[2], vertex_label= list(range(self.len_A + self.len_B)))

    #Create the titles
    axes[0].set_title("Graph A")
    axes[1].set_title("Graph B")
    axes[2].set_title("Graph AB")

    plt.show()
  
  def plot_heat_map(self):
    """
    Plots the heat map for the manifold alignment. 
    """
    #Create the figure
    plt.figure(figsize=(8, 6))
    
    #Plot the heat map
    sns.heatmap(self.block, cmap='viridis', mask = (self.block > 50))

    #Add title and labels
    plt.title('Block Matrix')
    plt.xlabel('Graph A Vertex')
    plt.ylabel('Graph B Vertex')

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

        FOSCTTM_score, CE_score, rf_score = self.get_scores(labels, n_comp = n_comp)

        print(f"RF score on full embedding: {rf_score}")
        print(f"Cross Embedding score: {CE_score}")
        print(f"Fraction of Samples Closest to thier Match: {FOSCTTM_score}")

        if type(labels) == type(None):
            #Set all labels to be the same
            labels = np.ones(shape = (len(self.emb)))

        #Time the plotting creation
        self.print_time()

        #Create styles to change the points from graph 1 to be triangles and circles from graph 2
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
        ax.set_title("SPUD", fontsize = 25)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
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
            sns.scatterplot(x = np.array(self.emb[self.known_anchors_adjusted, 0]).flatten(), y = np.array(self.emb[self.known_anchors_adjusted, 1]).flatten(), style = styles2,  linewidth = 2, markers= {"Domain A": "x", "Domain B" : "+"}, s = 45, color = "black", **kwargs)

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
            ax.set_title("Predicted Labels")

            plt.show()

        self.print_time("Time it took complete the plots: ")

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
        #Convert to a MDS
        self.print_time()
        mds = MDS(metric=True, dissimilarity = 'precomputed', n_components= n_comp, random_state = self.kwargs["random_state"])
        self.emb = mds.fit_transform(self.block)
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
    if np.issubdtype(first_labels[0].dtype, np.integer):
        rf_class = RFGAP(prediction_type="classification", y=labels, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=True, oob_score = True)
        if self.verbose > 1:
           print("RF-GAP score is accuracy")
    else:
        rf_class = RFGAP(prediction_type="regression", y=labels, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=True, oob_score = True)
        if self.verbose > 1:
           print("RF-GAP score is R^2")

    #Fit it for Data A and get proximities
    rf_class.fit(self.emb, y = labels)

    #Calculate FOSCTTM score
    try:    
        FOSCTTM_score = self.FOSCTTM(self.block[self.len_A:, :self.len_A])
    except: #This will run if the domains are different shapes
        FOSCTTM_score = None

    return FOSCTTM_score, CE_score, rf_class.oob_score_