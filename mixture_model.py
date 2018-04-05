import numpy as np
from scipy.stats import multivariate_normal

class MixtureModel:
    """
    The Gaussian Mixture model.
    
    The model estimates the parameters of a mixture of Gaussian distributions.
    
    The model is based on Expectation–Maximization (EM) algorithm.
    
    """
    def __init__(self, n_components, diag=False, min_deviation=5, adaptive=False, num_objects=250):
        """
        Parameters:
        ---------------
        n_components: int
            The number of components in mixture model

        diag: bool
            If diag is True, covariance matrix is diagonal
        
        min_deviation: int
            The lower bound of deviations (used not to get a degenerate distribution)
        
        """
        self.n_components = n_components  
        self.diag = diag
        self.min_deviation = min_deviation
        
    def _E_step(self, data):
        """
        E-step of the algorithm
        
        Parameters:
        ---------------
        data: numpy array shape (n_samples, n_features)
            Array of data points. Each row corresponds to a single data point.
        """    

        eps = 1e-100
        if self.diag:
            # not implemented yet
            pass
        else:
            self.probs = np.array([multivariate_normal.pdf(data, mean=self.Mean[i], cov=self.Sigma[i], allow_singular=True) 
                                   for i in range(self.n_components)]).T
            self.q_z = ((self.probs * self.w[np.newaxis, :] + eps / self.n_components) / 
                        (np.dot(self.probs, self.w)[:, np.newaxis] + eps))

    def _M_step(self, data):
        """
        M-step of the algorithm
        
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
            Array of data points. Each row corresponds to a single data point.
        """
        N, d = data.shape
        eps = 1e-100
        
        if self.diag:
            # not implemented yet
            pass
        else:
            q_z_sum = self.q_z.sum(axis=0)
            self.w = q_z_sum / N
            self.Mean = np.dot(self.q_z.T, data) / (q_z_sum[:, np.newaxis] + eps)
            self.Sigma = []
            for i in range(len(self.Mean)):
                data_centered = data - self.Mean[i][np.newaxis, :]
                Sigma = np.dot((data_centered * self.q_z.T[i][:, np.newaxis]).T, data_centered) / (q_z_sum[i] + eps)
                mask = np.diag(Sigma) < (self.min_deviation ** 2)
                Sigma[mask, mask] = self.min_deviation ** 2
                self.Sigma.append(Sigma)
            self.Sigma = np.array(self.Sigma)
    
    def EM_fit(self, data, max_iter=10, tol=1e-3, num_init_objects=5, seed=123,
               w_init=None, m_init=None, s_init=None, trace=False):
        """
        Parameters:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.

        max_iter: int
        Maximum number of EM iterations

        tol: int
        The convergence threshold

        w_init: numpy array shape(n_components)
        Array of the each mixture component initial weight

        Mean_init: numpy array shape(n_components, n_features)
        Array of the each mixture component initial mean

        Sigma_init: numpy array shape(n_components, n_features, n_features)
        Array of the each mixture component initial covariance matrix
        
        trace: bool
        If True then return list of likelihoods
        """

        N, d = data.shape
        self.q_z = np.zeros((N, self.n_components))
        self.tol = tol
        np.random.seed(seed)
        
        if w_init is None:
            self.w = np.ones(self.n_components) * (1 / self.n_components)
        else:
            self.w = w_init

        if m_init is None:
            indexes = np.random.choice(len(data), (self.n_components, num_init_objects))
            self.Mean = data[indexes, :].mean(axis=1)
        else:
            self.Mean = m_init
            
        if s_init is None:
            self.Sigma = np.array([np.identity(d) for i in range(self.n_components)])
        else:
            self.Sigma = s_init
        
        log_likelihood_list = []
        log_prev = np.inf
        
        for i in range(max_iter):
            # Perform E-step 
            self._E_step(data)
            
            # Compute loglikelihood
            log_likelihood_list.append(self.compute_log_likelihood(data))
            if (abs(log_likelihood_list[-1] - log_prev) < tol):
                break
            log_prev = log_likelihood_list[-1]
            # Perform M-step
            self._M_step(data)
        
        # Perform E-step
        self._E_step(data)
        # Compute loglikelihood
        log_likelihood_list.append(self.compute_log_likelihood(data))
        
        if trace:
            return self.w, self.Mean, self.Sigma, log_likelihood_list
        else:
            return self.w, self.Mean, self.Sigma
    
    def EM_with_different_initials(self, data, n_starts=3, max_iter=10, num_init_objects=5, seed=123,
                                   tol=1e-3, trace=False):
        """
        Parameters:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.

        n_starts: int
        The number of algorithm running with different initials

        max_iter: int
        Maximum number of EM iterations

        tol: int
        The convergence threshold

        Returns:
        --------
        Best values for w, Mean, Sigma parameters
        """
        np.random.seed(seed)
        best_w, best_Mean, best_Sigma, best_trace, max_log_likelihood = None, None, None, None, -np.inf
        seeds = np.random.randint(0, 1000, n_starts)
        for i in range(n_starts):
            w, Mean, Sigma, log_list = self.EM_fit(data, max_iter=max_iter, tol=tol, num_init_objects=num_init_objects,
                                                   trace=True, seed=seeds[i])
            if max_log_likelihood < log_list[-1]:
                best_w = w
                best_Mean = Mean
                best_Sigma = Sigma
                if trace:
                    best_trace = log_list
        
        self.w = best_w
        self.Mean = best_Mean
        self.Sigma = best_Sigma
        
        if trace:
            return self.w, self.Mean, self.Sigma, best_trace
        else:
            return self.w, self.Mean, self.Sigma     
        
    def compute_log_likelihood(self, data):
        """
        Parameetrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.
        """
        eps = 1e-100
        return np.log(np.dot(self.probs, self.w) + eps).sum()
