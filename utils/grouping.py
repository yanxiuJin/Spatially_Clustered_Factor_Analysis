import numpy as np
from sklearn.cluster import KMeans

class GroupingProcessor:
    """
    A class to split a DataFrame into groups based on different methods.
    """

    def __init__(self, data,sp,n):
        """
        Initialize the GroupingProcessor with the input DataFrame.

        Parameters:

        - data: The input DataFrame to be split.
        - n: The number of groups to split the indices into.
        """
        self.data = data
        self.sp = sp
        self.n = n

    def random_groups(self,random_seed=None):
        '''
        Randomly split the dataframe into n groups.
        
        Parameters:
        - scaled_df: The input DataFrame to be split.
        - n: The number of groups to split the dataframe into.

        Returns:
        - A list containing n random group labels.
        '''
        if random_seed is not None:
            np.random.seed(random_seed)
            group_label = np.random.choice(self.n, len(self.data))            
        else:
            group_label = np.random.choice(self.n, len(self.data))

        return group_label

    def kmeans_groups(self,random_seed=None):
        """
        Split the dataframe into n groups using K-means clustering.

        Parameters:
        - df: The input DataFrame to be split.
        - n: The number of groups to split the dataframe into.

        Returns:
        - A list of n lists, each containing indices of the dataframe that belong to each cluster.
        """

        # Apply KMeans clustering
        if random_seed is not None:
            kmeans = KMeans(n_clusters=self.n,init='k-means++',n_init=10,random_state=random_seed)
        else:
            kmeans = KMeans(n_clusters=self.n,init='k-means++',n_init=10)
        kmeans.fit( self.sp[:,-2:])
        sorted_centers = np.argsort(kmeans.cluster_centers_.sum(axis=1))
        new_labels = np.zeros_like(kmeans.labels_)
        for i, c in enumerate(sorted_centers):
            new_labels[kmeans.labels_ == c] = i
        # Retrieve cluster labels
        clusters = new_labels

        # Create a list of lists for indices in each cluster
        groups_dict = {key:[] for key in range(self.n)}
        for index, cluster in enumerate(clusters):
            groups_dict[cluster].append(index)

        return clusters



   