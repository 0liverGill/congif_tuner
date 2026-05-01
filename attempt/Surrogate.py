import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Optional







import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Optional


class RandomForestSurrogate:
    """Random Forest surrogate model."""

    def __init__(
        self,
        n_trees: int = 100,
        min_samples_leaf: int = 3,
        random_state: Optional[int] = None,
    ):
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None

    def fit(self, configuration: np.ndarray, performance: np.ndarray):
        self.model = RandomForestRegressor(
            n_estimators=self.n_trees,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=True,
            random_state=self.random_state,
        )
        self.model.fit(configuration, performance)

    def guess(self, configuration: np.ndarray):
        predictions = np.array(
            [tree.predict(configuration) for tree in self.model.estimators_]
        )
        mean = predictions.mean(axis=0)

        deviation = predictions.std(axis=0)
        deviation = np.maximum(deviation, 1e-8)
        return mean, deviation



"""
class RandomForestSurrogate():

    def __init__(
        self,
        n_trees: int = 100,
        min_samples_leaf: int = 3,
        random_state: Optional[int] = None,
    ):
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state


        self.model: Optional[RandomForestRegressor] = None
        

        def fit(self, configuration: np.ndarray ,performance: np.ndarray):

                self.model = RandomForestRegressor(
                n_estimators=self.n_trees,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                random_state=self.random_state,
                )
                self.model.fit(configuration, performance)
        

        def guess(self, configuration: np.ndarray):
             
            predictions = np.array([self.tree.predict(configuration) for tree in self.model.estimators])
            mean = predictions.mean(axis=0)
            deviation = predictions.mean(axis=0)
            #incase deviation is zero (divide by zero)
            deviation = np.maximum(deviation, 1e-8)

            return mean, deviation


"""



