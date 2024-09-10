import numpy as np

class Evaluation:
    def __init__(self, data, result):
        self.data = data
        self.factor_number = result.Load_Matrix_dict[1].shape[1]
        self.group_size = len(result.Load_Matrix_dict)
        self.likelihood = result.log_like_sum

    def calculate_AIC(self):
        num_feature = self.data.shape[1]
        num_params = (num_feature * self.factor_number + num_feature)*self.group_size
        aic = 2 * num_params - 2 * self.likelihood
        return aic

    def calculate_BIC(self):
        print()
        num_feature= self.data.shape[1]
        print()
        num_params = (num_feature * self.factor_number + num_feature)*self.group_size    
        bic = np.log(self.data.shape[0]) * num_params - 2 * self.likelihood
        return bic

