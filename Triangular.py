# Python Triangular methods
import numpy as np

def get_triangular(matrix, tol=1e-4, force_conversion = False):
    """If the matrix is symmetric, the function seeks to save memory by cutting out information that is
    redundant. 
    """
    #Check if the matrix is symetric
    if np.allclose(matrix, matrix.T, atol=tol):
      
      #flatten the array and just take the upper triangular part
      return matrix[np.triu_indices_from(matrix)]
    
    else:

      if force_conversion:
         #Make the matrix symmetric
         matrix = (matrix.T + matrix)/2

         return matrix[np.triu_indices_from(matrix)]

      #Return the original matrix if the matrix is not symmetric and we aren't forcing conversion
      return matrix

def index_triangular(upper_triangular, columns, return_indices=False):
    """Indexes the triangular matrix. If rows or columns are set to None, it returns all of them.
    If return_indices is True, it also returns the indices and the mask used for indexing."""

    # Check to see if the ndim = 1, else it's already built
    if upper_triangular.ndim != 1:
        
        upper_triangular = get_triangular(upper_triangular, force_conversion = True)

    # Get the size of the original matrix
    size = int((-1 + np.sqrt(1 + (8 * upper_triangular.size))) // 2)
    indices = np.triu_indices(size)

    # Create the mask for the specified columns
    col_mask = np.isin(indices[1], columns)

    # Create the mask for the symmetric counterparts in the lower triangle
    row_mask = np.isin(indices[0], columns)

    # Combine the row_mask with the off_diagonal_mask
    row_mask = row_mask & (indices[0] != indices[1])


    # Apply the combined mask to indices
    indices = np.concatenate((indices[0][col_mask], indices[1][row_mask]))
    upper_triangular = np.concatenate((upper_triangular[col_mask], upper_triangular[row_mask]))

    if return_indices:
        # Return the matrix, its indices, and mask
        return upper_triangular, indices
    else:
        # Just return the matrix
        return upper_triangular
    
def get_triangular_mean(upper_triangular, indices):
    """Calculates the mean of the upper-triangle in a highly efficient and vectorized fashion. 
      Column can be set to True to calculate the mean of the column or False for rows"""
    
    #Incase some indicies were skipped. TODO: upgrade this for when the labels aren't continuous
    indices -= indices.min()

    # Calculate the sums and counts for each index
    sums = np.bincount(indices, weights=upper_triangular)
    counts = np.bincount(indices)

    return sums / counts
  
def min_bincount(values, indices):
    # Get the unique indices and their positions
    unique_indices= np.unique(indices)
    
    # Initialize an array to store the minimum values and their positions
    min_values = np.full(unique_indices.shape, np.inf)
    
    # Use np.minimum.at to update the min_values array
    np.minimum.at(min_values, indices, values)

    # Create a mask for the minimum values
    min_pos = np.where((values == min_values[indices]))[0]

    # Extract the relevant indices
    list_thing = indices[min_pos]

    # Use numpy's unique function to find duplicates
    _, unique_indices = np.unique(list_thing, return_index=True)

    # Remove duplicates by selecting only the unique indice
    return min_pos[unique_indices]

def reconstruct_symmetric(upper_triangular):
    """Rebuilds the triangular to a symmetric graph"""

    #Check to see if the ndim = 1, else its already built
    if upper_triangular.ndim == 1:

      #Cache size for faster processing
      size = int((-1 + np.sqrt(1 + (8 * upper_triangular.size))) // 2)

      #Create a matrix filled with zeros
      matrix = np.zeros((size, size))

      #Get the indicies for the upper trianglar
      indices = np.triu_indices(size)

      #Reset the matrix
      matrix[indices] = upper_triangular

      # Mirror the upper part to the lower part
      matrix[(indices[1], indices[0])] = upper_triangular 

      return matrix
    
    else:
       return upper_triangular
    