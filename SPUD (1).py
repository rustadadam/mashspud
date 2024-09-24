#Shortest Path to Union Domains (SPUD)

"""
Adam's Notes
------------
Parameters to delete:
If we can show that mean does always better than abs, we can combine the OD_method and the similarity_measures, and greatly simplify the parameterization.

Tasks: Go through code and check where we want to make things triangular, and where to test their speeds in computing. 
3. Do I need the check for symmetric? IT always will be, unless given a precomputed data... ? No it will always be
5. Make the use-kernals thing automatic?
"""

#Install the libraries
from scipy.spatial.distance import pdist, squareform, _METRICS
import graphtools
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from Triangular import *

#Not necessary libraries for the minimal function
from time import time
import seaborn as sns

class SPUD:
  def __init__(self, distance_measure_A = "euclidean", distance_measure_B = "euclidean", knn = 5,
               OD_method = "default", agg_method = "log", IDC = 1, #TODO: See if its possible to get rid of either OD_method or similarity Measure
               similarity_measure = "default", #Maybe name this method and then have Jaccard, NAMA, and SPUD be seperate choosable methods
               float_precision = np.float32, verbose = 0, **kwargs):
        '''
        Creates a class object. 
        
        Arguments:
          :distance_measure_A: Either a function, "precomputed" or SciKit_learn metric strings for domain A. If it is a function, then it should
            be formated like my_func(data) and returns a distance measure between points. Self will be passed as the first argument.
            If set to "precomputed", no transformation will occur, and it will apply the data to the graph construction as given. The graph
            function uses Euclidian distance, but this may manually changed through kwargs assignment.

          :distance_measure_B: Either a function, "precomputed" or SciKit_learn metric strings for domain B. If it is a function, then it should
            be formated like my_func(data) and returns a distance measure between points. Self will be passed as the first argument.
            If set to "precomputed", no transformation will occur, and it will apply the data to the graph construction as given. The graph
            function uses Euclidian distance, but this may manually changed through kwargs assignment.

          :Knn: states how many nearest neighbors we want to use in the graph construction. If
            Knn is set to "connect" then it will ensure connection in the graph.

          :OD_method: stands for Off-diagonal method. Can be the strings "abs", "mean" or "default". "Abs" calculates the absolute distances between the
            shortest paths to the same anchor, where as default calculates the shortest paths by traveling through an anchor. "Mean" calculates the average distance
            by going through each anchor.

          :agg_method: States the method of how we want to adjust the off-diagonal blocks in the alignment. 
            It can be 'sqrt', 'log', any float, or 'None'.
            If 'sqrt', it applies a square root function, and then transposes it to start at 0. Best for when domains aren't the same shape.
            If 'log', it applies a natural log, and then gets the distances between each point. Requires 1 to 1 correspondence.
            If 'None', it applies no additional transformation besides normalizing the values between 0 and 1.
            If given a float, it multiplies the off-diagonal block by the float value.  

          :IDC: stands for Inter-domain correspondence. It is the similarity value for anchors points between domains. Often, it makes sense
            to set it to be maximal (IDC = 1) although in cases where the assumptions (1: the corresponding points serve as alternative 
            representations of themselves in the co-domain, and 2: nearby points in one domain should remain close in the other domain) are 
            deemed too strong, the user may choose to assign the IDC < 1.

          :similarity_measure: Can be default, NAMA,  or Jaccard. Default uses the alpha decaying kernal to determine distances between nodes. Jaccard applies the jaccard similarity
            to the resulting graph. NAMA uses the original distances from the Pdist function. We recommend using NAMA for data that is very large. 
            
          :verbose: can be any float or integer. Determines what is printed as output as the function runs.

          :**kwargs: key word values for the graphtools.Graph function. 
          '''

        #Set the values
        self.distance_measure_A = distance_measure_A
        self.distance_measure_B = distance_measure_B
        self.verbose = verbose
        self.knn = knn
        self.agg_method = agg_method
        self.kwargs = kwargs
        self.IDC = IDC
        self.OD_method = OD_method.lower()
        self.similarity_measure = similarity_measure.lower()
        self.float_precision = float_precision

        #Set self.emb to be None
        self.emb = None

  def fit(self, dataA, dataB, known_anchors):
        '''
        Does the work to compute the manifold alignment using shortest path distances. 
        
        Parameters:
          :dataA: the data for domain A. 
          :dataB: the data for domain B. 
          :known_anchors: this represents the points in domain A that correlate to domain B. Should be in a list
            formated like (n, 2), where n is the number of points that correspond. For any nth position, the 0th 
            place represents the point in domain A and the 1st position represents the point in domain B. Thus
            [[1,1], [4,3], [7,6]] would be appropiate.
        '''

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

        #If these parameters are true, we can skip this all:
        if self.OD_method != "default"  and self.similarity_measure == "nama":
           if self.verbose > 0:
              print("Skipping graph creating. Performing nearest anchor manifold alignment (NAMA) instead of SPUD.")

        else:
          #Create Igraphs and kernals from the input.
          self.print_time()
          self.graphA = graphtools.Graph(reconstruct_symmetric(self.distsA), knn = self.knn, knn_max= self.knn, **self.kwargs)
          self.graphB = graphtools.Graph(reconstruct_symmetric(self.distsB), knn = self.knn, knn_max= self.knn, **self.kwargs)

          self.kernalsA = get_triangular(self.graphA.K.toarray())
          self.kernalsB = get_triangular(self.graphB.K.toarray())

          self.graphA = self.graphA.to_igraph()
          self.graphB = self.graphB.to_igraph()
          self.print_time("Time it took to execute graphtools.Graph functions:  ")

        #Merge the graphs
        if self.OD_method == "default":
          if self.verbose > 0 and self.len_A > 1000:
             print("  --> Warning: Computing off-diagonal blocks will be exspensive. Consider setting OD_method to 'mean' or 'abs' for faster computation time.")

          if self.similarity_measure != "default" and self.verbose > 0:
             print("Will not be using the similarity measure. It only applies when OD_method does not equal 'default'.")
             

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
  
  """<><><><><><><><><><><><><><><><><><><><>     HELPER FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
  def print_time(self, print_statement =  ""):
    """A function that times the algorithms and returns a string of how
    long the function was last called."""

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
    """Normalizes the value to be between 0 and 1 and resets infinite values."""

    #Scale it and check to ensure no devision by 0
    if np.max(value[~np.isinf(value)]) != 0:
      value = (value - value.min()) / (value[~np.isinf(value)].max() - value.min()) 

    #Reset inf values
    value[np.isinf(value)] = 1

    return value

  def get_SGDM(self, data, distance_measure):
    """SGDM - Same Graph Distance Matrix.
    This returns the normalized distances within each domain."""

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

  """<><><><><><><><><><><><><><><><><><><><>     EVALUATION FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
  def cross_embedding_knn(self, embedding, Labels, knn_args = {'n_neighbors': 4}):
      """
      Returns the classification score by training on one domain and predicting on the the other.
      This will test on both domains, and return the average score.
      
      Parameters:
        :embedding: the manifold alignment embedding. 
        :Labels: a concatenated list of labels for domain A and labels for domain B
        :knn_args: the key word arguments for the KNeighborsClassifier."""

      (labels1, labels2) = Labels

      n1 = len(labels1)

      #initialize the model
      knn = KNeighborsClassifier(**knn_args)

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

        :off_diagonal: should be either off-diagonal portion (that represents mapping from one domain to the other)
        of the block matrix. 
        """
        n1, n2 = np.shape(off_diagonal)
        if n1 != n2:
            raise AssertionError('FOSCTTM only works with a one-to-one correspondence. ')

        dists = off_diagonal

        nn = NearestNeighbors(n_neighbors = n1, metric = 'precomputed')
        nn.fit(dists)

        _, kneighbors = nn.kneighbors(dists)

        return np.mean([np.where(kneighbors[i, :] == i)[0] / n1 for i in range(n1)])
  
  def compute_scores(self, labels, **kwargs):
      """Returns the FOSCTTM and CE score. 
      
      Labels should be the labels for domain A concatenated with domain B.
      
      Other key word arguments are to fit the MDS if necessary"""

      #Calculate FOSCTTM
      try:
        FOSCTTM_score = self.FOSCTTM(self.block[self.len_A:, :self.len_A])
      except:
        raise RuntimeError("SPUD must be fit first.")
      
      if self.verbose > 1:
        print(f"FOSCTTM: {FOSCTTM_score}") #This gets the off-diagonal part

      #Check to see if we already have created our embedding, else create the embedding.
      if type(self.emb) == type(None):
        #Time the embedding creation
        self.print_time()

        #Create the mds object and then the embedding
        mds = MDS(metric=True, dissimilarity = 'precomputed', **kwargs)
        self.emb = mds.fit_transform(self.block) 

        if self.verbose > 2:
          print("Embedding Calculated. Will not need to be calculated for future plotting again.")

        self.print_time("Time it took to calculate the embedding: ")
      
      elif self.verbose > 2:
         print("Embedding already calculated. ")

      #Seperate the labels into their respective domains
      first_labels = labels[:self.len_A]
      second_labels = labels[self.len_A:]

      #Compute Cross Embedding
      CE_score = self.cross_embedding_knn(self.emb, (first_labels, second_labels), knn_args = {'n_neighbors': 5})
      if self.verbose > 1:
        print(f"Cross Embedding: {CE_score}")

      return FOSCTTM_score, CE_score


  """<><><><><><><><><><><><><><><><><><><><>     PRIMARY FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
  def merge_graphs(self):
        """
        Creates a new graph from graphs A and B creating edges between corresponding points
        using the known anchors.
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
    """

    #The algorithm uses the kernals for speed and efficiency (so we don't waste time calculating similarities twice.)
    if self.verbose > 2:
       print(f"Preforming {self.OD_method} calculations.\n")

    #Set the matrix domains to equal the jaccard similarity
    if self.similarity_measure == "jaccard":
       matrixA = 1 - get_triangular(np.array(self.graphA.similarity_jaccard(pairs=None), dtype = self.float_precision))
       matrixB = 1 -  get_triangular(np.array(self.graphB.similarity_jaccard(pairs=None), dtype = self.float_precision))
    
    #Set the matrix domains to equal the distances via shortest paths
    elif self.similarity_measure == "distances" or self.similarity_measure == "default":
       matrixA = get_triangular(self.normalize_0_to_1(np.array(self.graphA.distances(weights = "weight"), dtype = self.float_precision)))
       matrixB = get_triangular(self.normalize_0_to_1(np.array(self.graphB.distances(weights = "weight"), dtype = self.float_precision)))
  
    #Just use the pure distance measure
    elif self.similarity_measure == "nama":
      matrixA = self.distsA
      matrixB = self.distsB

    elif self.similarity_measure == "kernals":
      matrixA = 1 - self.kernalsA
      matrixB = 1 - self.kernalsB

      #Change the interclass distances to be the kernals
      self.distsA = matrixA
      self.distsB = matrixB

    else: 
      raise RuntimeError("Did not understand the similarity measure. Please use 'distances' (default), 'nama', 'kernals', or 'jaccard'")

    #Perform the absolute value method
    if self.OD_method == "abs":

      #Subset A and B to only the columns so we only have the distances to the anchors
      anchor_dists_A, indiciesA = index_triangular(matrixA, columns = self.known_anchors[:, 0], return_indices=True)
      anchor_dists_B, indiciesB = index_triangular(matrixB, columns = self.known_anchors[:, 1], return_indices=True)

      # Find the indices of the closest anchors for each node in both graphs
      A_smallest_index = min_bincount(anchor_dists_A, indiciesA) #NOTE: These are the index positions of the smallest value, but its flattened according to the triangular. 
      B_smallest_index = min_bincount(anchor_dists_B, indiciesB) #NOTE Continued: If the triangulars are different sizes we might need to adjust the values because they wont match up. 

      #Strecth A and B to be the correct sizes, and then select the subtraction anchors
      matrixA = np.repeat(anchor_dists_A[A_smallest_index].astype(self.float_precision), repeats=self.len_B)
      matrixB = np.tile(anchor_dists_B[A_smallest_index].astype(self.float_precision), self.len_A)

      off_diagonal_using_A_anchors = np.abs(matrixA - matrixB)

      #Strecth A and B to be the correct sizes, and then select the subtraction anchors
      matrixA = np.repeat(anchor_dists_A[B_smallest_index].astype(self.float_precision), repeats=self.len_B)
      matrixB = np.tile(anchor_dists_B[B_smallest_index].astype(self.float_precision), self.len_A)

      off_diagonal_using_B_anchors = np.abs(matrixA - matrixB)

      #Perform the calculation
      off_diagonal = np.reshape(np.minimum(off_diagonal_using_A_anchors, off_diagonal_using_B_anchors), newshape=(self.len_A, self.len_B))

    elif self.OD_method == "mean":

      #Take the mean of each one first, then select.
      anchor_dists_A = get_triangular_mean(*index_triangular(matrixA, columns = self.known_anchors[:, 0], return_indices=True)) 
      anchor_dists_B = get_triangular_mean(*index_triangular(matrixB, columns = self.known_anchors[:, 1], return_indices=True))

      #Strecth A and B to be the correct sizes, and so each value matches up with each other value.
      anchor_dists_A = np.repeat(anchor_dists_A.astype(self.float_precision), repeats= self.len_B)
      anchor_dists_B = np.tile(anchor_dists_B.astype(self.float_precision), self.len_A)
      
      #Convert it to the square matrix. NOTE: Do we want to convert it back into a triangular????? - Probably not.
      off_diagonal = np.reshape(np.abs(anchor_dists_A - anchor_dists_B), newshape=(self.len_A, self.len_B))

    else:
       raise RuntimeError("Did not understand your input for OD_method (Off-Diagonal method). Please use 'mean', 'abs', or 'default'.")
             
    return off_diagonal

  def get_block(self):
    """
    Returns a transformed and normalized block.
    
    Parameters:
      :graph: should be a graph that has merged together domains A and B.
      
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

  """<><><><><><><><><><><><><><><><><><><><>     VIZUALIZATION FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
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
        """A useful visualization function to veiw the embedding.
        
        Arguments:
            :labels: should be a flattened list of the labels for points in domain A and then domain B. 
                If set to None, the cross embedding can not be calculated, and all points will be colored
                the same. 
            :n_comp: The amount of components or dimensions for the MDS function.
            :show_lines: should be a boolean value. If set to True, it will plot lines connecting the points 
                that correlate to the points in the other domain. It assumes a 1 to 1 correpondonce. 
            :show_anchors: should be a boolean value. If set to True, it will plot a black square on each point
                that is an anchor. 
            :**kwargs: additional key word arguments for sns.scatterplot function.
        """

        #Check to see if we already have created our embedding, else create the embedding.
        if type(self.emb) == type(None):
          #Time the embedding creation
          self.print_time()

          #Create the mds object and then the embedding
          mds = MDS(metric=True, dissimilarity = 'precomputed', **kwargs)
          self.emb = mds.fit_transform(self.block) 

          if self.verbose > 2:
            print("Embedding Calculated. Will not need to be calculated for future plotting again.")

          self.print_time("Time it took to calculate the embedding: ")

        #Check to make sure we have labels
        if type(labels)!= type(None):
            #Seperate the labels into their respective domains
            first_labels = labels[:self.len_A]
            second_labels = labels[self.len_A:]

            #Calculate Cross Embedding Score
            try: #Will fail if the domains shapes aren't equal
                print(f"Cross Embedding: {self.cross_embedding_knn(self.emb, (first_labels, second_labels), knn_args = {'n_neighbors': 5})}")
            except:
                print("Can't calculate the Cross embedding")
        else:
            #Set all labels to be the same
            labels = np.ones(shape = (len(self.emb)))

        #Calculate FOSCTTM Scores
        try:    
            print(f"FOSCTTM: {self.FOSCTTM(self.block[self.len_A:, :self.len_A])}") #This gets the off-diagonal part
        except: #This will run if the domains are different shapes
            print("Can't compute FOSCTTM with different domain shapes.")


        #Time the plotting creation
        self.print_time()

        #Create styles to change the points from graph 1 to be triangles and circles from graph 2
        styles = ['Domain A' if i < self.len_A else 'Domain B' for i in range(len(self.emb[:]))]

        #Create the figure
        plt.figure(figsize=(14, 8))

        #Imporrt pandas for the categorical function
        from pandas import Categorical

        #If show_pred is chosen, we want to show labels in Domain B as muted
        if show_pred:
            ax = sns.scatterplot(x = self.emb[self.len_A:, 0], y = self.emb[self.len_A:, 1], color = "grey", s=120, marker= "o", **kwargs)
            ax = sns.scatterplot(x = self.emb[:self.len_A, 0], y = self.emb[:self.len_A, 1], hue = Categorical(first_labels), s=120, marker= "^", **kwargs)
        
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
            sns.scatterplot(x = np.array(self.emb[self.known_anchors_adjusted, 0]).flatten(), y = np.array(self.emb[self.known_anchors_adjusted, 1]).flatten(), style = styles2,  linewidth = 2, markers= {"Domain A": "x", "Domain B" : "+"}, s = 45, color = "black")

        #Show plot
        plt.show()

        #Show the predicted points
        if show_pred and type(labels) != type(None):

            #Instantial model, fit on domain A, and predict for domain B
            knn_model = KNeighborsClassifier(n_neighbors=4)
            knn_model.fit(self.emb[:self.len_A, :], first_labels)
            second_pred = knn_model.predict(self.emb[self.len_A:, :])
            
            #Create the figure
            plt.figure(figsize=(14, 8))

            #Now plot the points
            ax = sns.scatterplot(x = self.emb[:, 0], y = self.emb[:, 1], style = styles, hue = Categorical(np.concatenate([first_labels, second_pred])), s=120, markers= {"Domain A": "^", "Domain B" : "o"}, **kwargs)

            #Set the title
            ax.set_title("Predicted Labels")

            plt.show()

        self.print_time("Time it took complete the plots: ")
