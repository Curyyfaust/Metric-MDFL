import numpy as np
from typing import Union

def risk_comp(forecast, real, aggregate: bool = True,
              under_penalty: float = 5.0, over_penalty: float = 50.0) -> Union[np.ndarray, float]:
    """
    Compute risk penalties for forecasts vs. real values.

    Rules (element-wise):
      - if forecast < real: penalty = 50 * (real - forecast)    (under-forecast, heavy penalty)
      - if forecast > real: penalty = 5  * (forecast - real)    (over-forecast, light penalty)  
      - if equal:           penalty = 0

    Parameters:
      - forecast, real: array-like (broadcastable to same shape)
      - aggregate: if True return scalar sum of penalties, else return array of penalties
      - under_penalty: penalty multiplier for under-forecasting (default 50.0)
      - over_penalty: penalty multiplier for over-forecasting (default 5.0)

    Returns:
      - float (sum) if `aggregate=True`, else np.ndarray of same shape as broadcasted inputs
    """
    f, r = np.broadcast_arrays(np.asarray(forecast, dtype=float),
                               np.asarray(real, dtype=float))
    penalties = np.zeros_like(r, dtype=float)
    mask_under = f < r   # under-forecast -> heavy penalty (missed opportunity)
    mask_over = f > r    # over-forecast -> light penalty (over-commit)
    penalties[mask_under] = under_penalty * (r[mask_under] - f[mask_under])
    penalties[mask_over] = over_penalty * (f[mask_over] - r[mask_over])
    return penalties.sum() if aggregate else penalties