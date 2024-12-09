import numpy as np
import pandas as pd

def get_matrix(X: pd.DataFrame) -> np.ndarray:
    """
    Function to obtain a matrix of Gower's distance between data points. Takes pandas dataframe object containing observation data and returns numpy array of pairwise Gower distances between points.
    
    :param: pd.DataFrame: Dataframe containing the observation data.

    :return: np.ndarray: Array with pairwise Gower distances between points.
    """
    # empty results dataframe of shape n_obs x n_obs
    distances = pd.DataFrame(0, index= range(X.shape[0]), columns= range(X.shape[0]))

    # loop over all combinations of points for distance calculation
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):

            # Only calculate distances when necessary for efficiency 
            if distances.iloc[i,j] == 0:
                print(f"Calculating dist for {i}, {j}")
                distances.iloc[i,j]  = calc_gower(X, i, j)
                distances.iloc[j,i] = distances.iloc[i,j] 
                print(f"Distances {i},{j} added")

            else:
                pass

    # return np.ndarray
    return distances.values

def calc_gower(X: pd.DataFrame, i: int, j: int) -> float:
    """
    Function to calculate Gower's distance between two points i and j from a dataset X.

    :param X: pd.DataFrame: Dataframe containing the observation data.
    :param i: int: index for point i
    :param j: int: index for point j

    :return: float: Gower distance between points i and j in data X.
    """

    # initialise empty lists for categorical and numerical value distances
    cat_dist = []
    num_dist = []

    # for every combination loop over all variables
    for k in range(X.shape[1]):
        
        # handle continuous and categorical variables and calc distances for each var
        # cat variables (str and bool) : intermediate dist = 1 if = ; 0 if different
        if type(X.iloc[i,k]) in [str, np.bool_]:
            cat_dist.append(np.where(X.iloc[i,k] == X.iloc[j,k], 1, 0))
        # numerical variables : 0 - 1 from least to most similar
        elif type(X.iloc[i,k]) in [np.int64, np.int32, np.float64]:
            num_dist.append(1-(abs(X.iloc[i,k] - X.iloc[j,k])) / (X[X.columns[k]].max() - X[X.columns[k]].min()))
        else:
            cat_dist.append(np.where(X.iloc[i,k] == X.iloc[j,k], 1, 0))

    # calculate distance between points, add to df

    distance_ij = 1 - ((sum(num_dist) + sum(cat_dist)) / len(X.columns))

    return distance_ij

if __name__ == '__main__':

    data = {'num_var1': list(range(1,6)),
        'num_var2': np.linspace(0, 1, 5),
        'cat_var2': ['one', 'two', 'three', 'four', 'five'],
        'bool_var3': [i % 2 == 0 for i in range(5)]}
    
    X = pd.DataFrame(data)
    d = get_matrix(X)

    print(d)

    #data = {'num_var1': list(range(1,101)),
        #'num_var2': np.linspace(0, 1, 100),
        #'cat_var2': ['one', 'two', 'three', 'four', 'five']*20,
        #'bool_var3': [i % 2 == 0 for i in range(100)]}    

    #X = pd.DataFrame(data)
    #d = get_matrix(X)

    #print(d)
    #print(d.shape)

