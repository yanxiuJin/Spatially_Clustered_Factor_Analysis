from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import numpy as np
import matplotlib.pyplot as plt

class FactorModel:
    def __init__(self,data) :
        self.data = data

    def factor_analysis_Check(self):
        
        kmo = calculate_kmo(self.data)
        bartlett = calculate_bartlett_sphericity(self.data)

        if kmo[1]>0.5 and bartlett[1] <0.05:
            print(f'KMO:{kmo[1]}, Bartlett:{bartlett[1]}. Suitable for Factor Analysis!')
        else:
            print(f'KMO:{kmo[1]}, Bartlett:{bartlett[1]}. Warning: NOT suitable for Factor Analysis!')

    def factor_number(self,K=100):
        '''Parallel Analysis to determine the number of factors'''


        ################
        # Create a random matrix to match the dataset
        ################
        n, m = self.data.shape
        # Set the factor analysis parameters
        fa = FactorAnalyzer(n_factors=1, method='minres')
        # Create arrays to store the values
        sumFactorEigens = np.empty(m)
        # Run the fit 'K' times over a random matrix
        for runNum in range(0, K):
            fa.fit(np.random.normal(size=(n, m)))
            sumFactorEigens = sumFactorEigens + fa.get_eigenvalues()[1]
        # Average over the number of runs
        avgFactorEigens = sumFactorEigens / K

        ################
        # Get the eigenvalues for the fit on supplied data
        ################
        fa.fit(self.data)
        dataEv = fa.get_eigenvalues()

        # Find the suggested stopping points
        suggestedFactors = sum((dataEv[1] - avgFactorEigens) > 0)
        print('Parallel analysis suggests that the number of factors = ', suggestedFactors)

        return int(suggestedFactors)
    
    def ParallelAnalysis(self, K=100, printEigenvalues=False):
        ################
        # Create a random matrix to match the dataset
        ################
        n, m = self.data.shape
        # Set the factor analysis parameters
        fa = FactorAnalyzer(n_factors=1, method='minres')
        # Create arrays to store the values
        sumFactorEigens = np.empty(m)
        # Run the fit 'K' times over a random matrix
        for runNum in range(0, K):
            fa.fit(np.random.normal(size=(n, m)))
            sumFactorEigens = sumFactorEigens + fa.get_eigenvalues()[1]
        # Average over the number of runs
        avgFactorEigens = sumFactorEigens / K

        ################
        # Get the eigenvalues for the fit on supplied data
        ################
        fa.fit(self.data)
        dataEv = fa.get_eigenvalues()
        # Set up a scree plot
        plt.figure(figsize=(6, 4))

        ################
        ### Print results
        ################
        # Find the suggested stopping points
        suggestedFactors = sum((dataEv[1] - avgFactorEigens) > 0)
        print('Parallel analysis suggests that the number of factors = ', suggestedFactors )


        ################
        ### Plot the eigenvalues against the number of variables
        ################
        # Line for eigenvalue 1
        plt.plot([0, m+1], [1, 1], 'k--', label='Eigenvalue = 1',alpha=0.3)

        plt.plot(range(1, m+1), dataEv[1], '-bo', label='FA - data')
        plt.plot(range(1, m+1), avgFactorEigens, '--rx', label='FA - random', alpha=0.8)
        plt.title('Parallel Analysis Scree Plots', {'fontsize': 20})
        plt.xlabel('Factors', {'fontsize': 15})
        plt.xticks(ticks=range(1, m+1,2), labels=range(1, m+1,2))
        plt.ylabel('Eigenvalue', {'fontsize': 15})
        plt.grid('--', alpha=0.3)
        plt.legend()
        plt.show()


    def fit_fa(self,factors_n):
        fa = FactorAnalyzer(n_factors=factors_n,rotation='varimax',method='minres')
        fa.fit(self.data)
        Load_Matrix=fa.loadings_
        fa_score=fa.transform(self.data)
        variance_info = fa.get_factor_variance()
        mean = fa.mean_

        e = np.diag(np.var(self.data - np.dot(fa.loadings_, fa_score.T).T,axis=0))
        cov = np.dot(fa.loadings_,(fa.loadings_).T) + e
        cov_=(cov + cov.T) / 2


        return cov_,Load_Matrix,e,fa_score,variance_info,mean