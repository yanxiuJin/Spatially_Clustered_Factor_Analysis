import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import networkx as nx
from pathlib import Path
import os



class DataProcessor:
    def __init__(self,data) :
        self.data = data

    def varibles_filter(self,threshold =0.9):
        '''
        Remove the varibles with high correlation
        :param threshold: correlation value. Those greather than this value will be remove.
        '''
        numeric_cols = self.data.select_dtypes(include=['number'])# Select columns with numerical data types
        # Calculate the correlation matrix for the numeric columns
        corr_matrix = numeric_cols.corr().abs()
        # Find columns to drop based on the threshold
        cols_to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > threshold:
                    cols_to_drop.add(corr_matrix.columns[j])

            # Print removed columns (optional)
        print("Columns removed due to high correlation:", cols_to_drop)

        # Drop the columns from the DataFrame
        return numeric_cols.drop(columns=cols_to_drop)

    def Standardizing(self,filter=False):
        '''
        Standardize the data
        :param filter: If True, remove the varibles with high correlation.
        '''
        if filter:
            data_to_standardizer = self.varibles_filter()
        else:
            data_to_standardizer = self.data.select_dtypes(include=['number'])
        # Standardizing the data
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(data_to_standardizer), columns=data_to_standardizer.columns)
        return scaled_df
    
class SpatialMatrixProcessor:

    def __init__(self, sp_file, sp_mode , matrix_file, file_name):
        '''
        :param sp_file: The spatial data file.
        :param sp_mode: The mode to determine which matrix to use ('haversine', 'top5', or 'euclidean').
        :param matrix_file: Whether to read existing matrix file.
        :param file_name: The file name of matrix file.
        '''

        self.sp = sp_file
        self.sp_mode = sp_mode
        self.matrix_file = matrix_file
        current_file = Path(__file__)
        parent_folder = current_file.parent.parent
        self.file_path = os.path.join(parent_folder, 'sample_data', file_name)

    def haversine(self,lon1, lat1, lon2, lat2):
        '''
        Calculate the haversine distance between two points.

        :param lon1: The longitude of the first point.
        :param lat1: The latitude of the first point.
        :param lon2: The longitude of the second point.
        :param lat2: The latitude of the second point.

        :return: The haversine distance between the two points.
        '''
        # Haversine formula
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r
    
    def dis_matrix_haversine(self):
        '''
        Calculate the distance matrix using the haversine formula.

        :param lon_col: The name of the column containing the longitude values.
        :param lat_col: The name of the column containing the latitude values.

        :return: The distance matrix.
        '''
        lon = self.sp[:,-2]
        lat = self.sp[:,-1]
        # Initialize the distance matrix.
        n = len(self.sp) 
        distance_matrix = np.zeros((n, n))

        # Calculate the distance.
        for i in range(n):
            for j in range(i+1, n):
                dist = self.haversine(lon[i], lat[i], lon[j], lat[j])
                distance_matrix[i, j] = np.exp(-dist) # Exponential decay, the closer the points, the larger the value.
                distance_matrix[j, i] = np.exp(-dist)
        
        np.fill_diagonal(distance_matrix, np.min(distance_matrix)) # Set the diagonal to the minimum value to avoid 0s.

        return distance_matrix

    def dis_matrix_top5(self):
        '''
        Calculate the distance matrix using the haversine formula and the top 5 distances.
        
        :return: The distance matrix with the top 5 distances set to 1 and the rest to 0.
        '''

        # Calculate the distance matrix using the haversine formula
        distance_matrix = self.dis_matrix_haversine()
        # Sort the distances in descending order
        sorted_distances = np.sort(distance_matrix, axis=1)[:, ::-1]
        # Set the top 5 distances to 1 and the rest to 0
        distance_matrix_ = np.where(distance_matrix >= sorted_distances[:, 4].reshape(-1, 1), 1, 0)

        return distance_matrix_
    
    def dis_matrix_Euclidean(self):
        '''
        Calculate the distance matrix using the Euclidean distance.
        '''
        X = self.sp
        sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)  
        matrix = np.exp(-sq_dists / (2 * 0.1**2))

        return matrix
    

    def prepare_spatial_matrix(self):
        # Load the spatial matrix if it exists
        distance_matrix = None
        if self.matrix_file:
            distance_matrix = np.load(self.file_path)
            print("Spatial matrix loaded.")
        else:
            if self.sp_mode == 'haversine':
                distance_matrix = self.dis_matrix_haversine()
            elif self.sp_mode == 'top5':
                distance_matrix = self.dis_matrix_top5()
            elif self.sp_mode == 'euclidean':
                distance_matrix = self.dis_matrix_Euclidean()
            else:
                raise ValueError("Invalid spatial mode. Must be 'haversine', 'top5', or 'euclidean'.")

        return distance_matrix
       
    
    def calculate_spatial_weight(self,x, group_size, g_index, distance_matrix):
        '''
        Calculate the spatial weight for a given sample x based on the specified mode and weights.

        :x: The index of the sample.
        :g_index: The group index for the samples.

        :returns: The spatial weight for the sample x.
        '''
        group_sum = []
        for j in range(group_size):
            group_weight=[]
            for index in np.where(g_index == j)[0]:
                weight_value = distance_matrix[x, index]
                group_weight.append(weight_value)
    
            group_sum.append(sum(group_weight))
            
        return np.array(group_sum)
