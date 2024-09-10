###------------------------------------------------------------###
###  Functions for spatially clustered factor analysis (SCFA)  ###
###------------------------------------------------------------###

import numpy as np
import warnings
import copy
from collections import deque
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, './utils')
from data_processor import DataProcessor, SpatialMatrixProcessor
from statistical_tools import multivariate_likelihood, has_constant_columns
from grouping import GroupingProcessor
from factor_analysis import FactorModel


class ClusteredFAResult:
    def __init__(self, Load_Matrix_dict, Noise_Variance_dict, groups_indexs,Fa_scores,log_like_sum, g_count,initial_groups,k,Variance_infos,Means_dict):
        self.Load_Matrix_dict = Load_Matrix_dict
        self.Noise_Variance_dict = Noise_Variance_dict
        self.groups_indexs = groups_indexs
        self.log_like_sum = log_like_sum
        self.fa_scores = Fa_scores
        self.g_count = g_count
        self.variance_infos = Variance_infos
        self.means_dict = Means_dict
        self.initial_groups = initial_groups
        self.iteration = k


class ClusteredFactorAnalysis:
    def __init__(self, data, sp, group_size = None, factor_number = None, group_type = 'random', maxitr = 50, sp_mode = 'haversine', phi=1, matrix_file=False, file_name = 'distance_matrix_haversine.npy',multiple_try = False,random_state=None):
        ''' 
        :param data: The input DataFrame to be split.
        :param sp: The spatial data file.
        :param group_size: The number of groups to split the dataframe into.
        :param factor_number: The number of factors to be extracted.
        :param group_type: The type of grouping method to be used. "random" or "kmeans".
        :param maxitr: The maximum number of iterations to be performed.
        :param sp_mode: The spatial mode to be used. "haversine","top5","euclidean"
        :param phi: The spatial parameter to be used.
        :param matrix_file: The file path to the matrix file.
        :param file_name: The file name of the matrix file.
        :param multiple_try: Whether to perform multiple tries.
        :param random_state: The random state to be used.
        '''

        self.data = data
        self.sp = sp
        self.group_size = group_size
        self.factor_number = factor_number
        self.group_type = group_type
        self.maxitr = maxitr
        self.sp_mode = sp_mode
        self.phi = phi
        self.matrix_file = matrix_file
        self.file_name = file_name
        self.multiple_try = multiple_try
        self.random_state = random_state
        self.group_indices_queue = deque(maxlen=2) # queue to store the group indices of the last two iterations
        self.jump_count = 0  # initialize the number of jumps to 0


    def prepare_data(self):

        self.input_data =DataProcessor(self.data.select_dtypes(include=['number'])).Standardizing() 

        if self.factor_number is None:
            fa = FactorModel(self.input_data)
            self.factor_number = fa.factor_number() 

    def assign_groups(self):
        """Assign data to groups based on the specified grouping method.""" 
        self.prepare_data()       
        grp=GroupingProcessor(self.input_data,self.sp,self.group_size)
        if self.group_type == 'random':
            return grp.random_groups(self.random_state)
        elif self.group_type == 'kmeans':
            return grp.kmeans_groups() 
        else:
            raise ValueError("Invalid group type. Choose 'random' or 'kmeans'.")

    def detect_jumping_samples(self, g_index):
        
        current_group_indices = np.copy(g_index)

        if len(self.group_indices_queue) < 2:
            self.group_indices_queue.append(current_group_indices)
            return False

        prev2_group_indices = self.group_indices_queue[0]

        if np.array_equal(current_group_indices, prev2_group_indices):
            self.group_indices_queue.append(current_group_indices)
            return True
        else:
            self.group_indices_queue.append(current_group_indices)
            return False

    def group_factor_analysis(self,g_index,Load_Matrix_dict,Noise_Variance_dict,Cov_dict,Fa_scores,Variance_infos,Means_dict):
        # group_loglikes=[]
        for j in range(self.group_size):
            indices = np.where(g_index == j)[0]
            group = self.input_data.values[indices]
            if has_constant_columns(group):
                continue

            elif len(group) > self.factor_number+2: 
                try:
                    fa = FactorModel(group)
                    
                    cov, Load_Matrix, error, fa_score, variance_info,mean = fa.fit_fa(self.factor_number)
                    Load_Matrix_dict[j] = Load_Matrix
                    Noise_Variance_dict[j] = error
                    Cov_dict[j] = cov                  
                    Fa_scores[indices] = fa_score  
                    Variance_infos[j] = variance_info
                    Means_dict[j] = mean
         
                except np.linalg.LinAlgError:

                    continue


        return Load_Matrix_dict, Noise_Variance_dict, Cov_dict, Fa_scores, Variance_infos,Means_dict


    def Clustered_FA_single(self,distance_matrix,spm):

        #initialize factor analysis parameters
        X = self.input_data.values
        mean_vector = np.zeros(X.shape[1])
        g_count = {key: [np.sum(self.groups_indexs==key)] for key in range(self.group_size)}
        log_like_sum = []
        Noise_Variance_dict={i: np.zeros((X.shape[1], X.shape[1])) for i in range(self.group_size)}
        Load_Matrix_dict = {i: np.zeros((X.shape[1], self.factor_number)) for i in range(self.group_size)}
        Cov_dict = {i: np.zeros((X.shape[1], X.shape[1])) for i in range(self.group_size)}
        Fa_scores = np.zeros((X.shape[0], self.factor_number))
        Means_dict = {i: np.zeros(X.shape[1]) for i in range(self.group_size)}
        Variance_infos = {i: () for i in range(self.group_size)}
        g_index=copy.deepcopy(self.groups_indexs)

        for k in range(self.maxitr):
            Noise_Variance_dict_old=copy.deepcopy(Noise_Variance_dict)  
            Load_Matrix_dict, Noise_Variance_dict, Cov_dict, Fa_scores,Variance_infos,Means_dict= self.group_factor_analysis(g_index,Load_Matrix_dict,Noise_Variance_dict,Cov_dict,Fa_scores,Variance_infos,Means_dict)   
                   
            #calculate likelihood and regroup
            max_loglike=[]       

            for i in range(len(X)):
                val=[]
                for j in range(len(Cov_dict)):
                    val.append(multivariate_likelihood(X[i], mean_vector, Cov_dict[j]))

                if distance_matrix is not None:  
                    weight_sum = spm.calculate_spatial_weight(i,self.group_size, g_index, distance_matrix)
                    PenLoglike = np.array(val) + np.dot(self.phi,weight_sum)
                else:
                    PenLoglike = np.array(val)
                g_index[i] = np.argmax(PenLoglike)
                max_loglike.append(val[np.argmax(PenLoglike)])  
            
            log_like_sum = np.sum(max_loglike)

            #save group_change
            for q in range(self.group_size): 
                g_count[q].append(np.sum(g_index == q)) 
            
            # convergence check 
            if k> 0:
                N_diff=0
                for w in range(self.group_size):
                    N_diff= N_diff+np.sum(abs(Noise_Variance_dict[w]-Noise_Variance_dict_old[w]))/np.sum(Noise_Variance_dict_old[w])
                if N_diff<0.00001:
                    # print(f"Reaching Convergence after {k} iterations.")  
                    break   

            if self.detect_jumping_samples(g_index):
                self.jump_count += 1
                if self.jump_count >= 5:
                    # print("Jumping between groups detected 5 times. Stopping iteration.")
                    break

        return ClusteredFAResult(Load_Matrix_dict,Noise_Variance_dict,g_index,Fa_scores,log_like_sum,g_count,self.groups_indexs,k,Variance_infos,Means_dict)                              


    def Clustered_FA_multi(self,distance_matrix,spm,times=10):
        bic_old=100000000     
        for r in range(times):
            self.groups_indexs = self.assign_groups()
            results=self.Clustered_FA_single(distance_matrix,spm)
            bic = self.calculate_BIC(results.log_like_sum)
            if bic < bic_old:
                bic_old = bic
                best_re = results
        return best_re    
                   

    def Clustered_FA_no_group(self,distance_matrix,spm):      
        bic_old=100000000     
        size= (self.data.values.shape[0]//self.data.values.shape[1])//3   
        for s in range(2,size):

            self.group_size = s
            self.groups_indexs = self.assign_groups()  
            results=self.Clustered_FA_single(distance_matrix,spm)
            
            bic = self.calculate_BIC(results.log_like_sum)
            if bic < bic_old:
                bic_old = bic
                group_size = s
                best_re = results

        print("optimal group size is:"+str(group_size))
        return best_re     
    
    def fit(self):
        #initialize group and spatial matrix 

        spm= SpatialMatrixProcessor(self.sp, self.sp_mode,self.matrix_file,self.file_name)    
        distance_matrix = spm.prepare_spatial_matrix()

        if self.group_size is None:
            return self.Clustered_FA_no_group(distance_matrix=distance_matrix,spm=spm)
        elif self.multiple_try:
            return self.Clustered_FA_multi(distance_matrix=distance_matrix,spm=spm)
        else:
            #initialize group and spatial matrix
            self.groups_indexs = self.assign_groups()
            return self.Clustered_FA_single(distance_matrix=distance_matrix,spm=spm)
