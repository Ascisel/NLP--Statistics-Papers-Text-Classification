"""
Resampling methods implemented in the file help deal with imbalanced data.
Among the implemented methods there are: 
    * RandomOverSampler - Copy samples from minority class
"""

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from typing import Tuple


def resample_training_data(
    X_train: np.array, y_train: np.array
) -> Tuple[np.array, np.array]:
    """
    Random upsampling minor class

    Parameters:
    -----------
    X_train - training data
    y_train - training labels
    Returns:
    --------
    X_res - resampled training data
    y_res - resampled training labels
    """
    rn = RandomOverSampler(random_state=42)
    X_sm, y_sm = rn.fit_resample(X_train, y_train)
    return np.asarray(X_sm), np.asarray(y_sm)  
