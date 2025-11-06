# analyze.py

import numpy as np

def summarize(final_wealths: np.ndarray):
    """
    Summarize the final wealths from multiple simulation paths.

    :param final_wealths: A numpy array of final wealth values from simulations.
    :return: A dictionary with summary statistics.
    """
    return {
        'mean': float(np.mean(final_wealths)),
        'median': float(np.median(final_wealths)),
        'p05': float(np.percentile(final_wealths, 5)),
        'p95': float(np.percentile(final_wealths, 95)),
        'std': float(np.std(final_wealths))
    }