import numpy as np
from scipy.stats import norm


#https://schneppat.com/expected-improvement_ei.html for more info on how this works
class ExpectedImprovement():

    def __init__(self, ee: float = 0.01):
        """
        Args:
            ee: Exploration-exploitation trade-off parameter.
                default = 0.01 which is a mild exploration bonus
                higher values = more exploration
        """
        self.ee = ee    

    

    def evaluate(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        best_so_far: float,
    ) -> np.ndarray:
        """
        Compute Expected Improvement for each candidate.
        
        Args:
            mu: Predicted performance (lower = better for minimisation).
            sigma: Prediction uncertainty.
            best_so_far: Best (lowest) measured performance so far.
            
        Returns:
            ei: Expected Improvement scores. Shape (n_candidates,).
                Higher EI = more promising candidate.
        """
        # How much better than current best do we expect each candidate to be?
        improvement = best_so_far - mu + self.ee
        
        # Normalise by uncertainty to get Z-scores.
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(sigma > 1e-10, improvement / sigma, 0.0)
        
       
        ei = np.where(
            sigma > 1e-10,
            #IE formula
            improvement * norm.cdf(Z) + sigma * norm.pdf(Z),
            np.maximum(improvement, 0.0),
        )
        
        return ei    
    

