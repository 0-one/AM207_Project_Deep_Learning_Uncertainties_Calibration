import numpy as np
import pandas as pd


def generate_data(func, points, seed=0):
    """Generate a dataframe containing the covariate X, and observations Y

    The X's are generated uniformly over each of the supplied segments.

    Args:
        func: a scipy.stats function
        points: a list of dictionaries describing the points
            The expected format: [{"n_points": 10, "xlim": [-1, 1]}, ...]
        seed: random seed (default: {0})

    Returns:
        a pandas DataFrame with the generated X and Y
    """
    np.random.seed(seed)

    data = []
    for segment in points:
        x = np.linspace(*segment["xlim"], num=segment["n_points"])
        distribution = func(x)
        # Generate observations
        y = distribution.rvs()
        df = pd.DataFrame({"x": x, "y": y})
        data.append(df)

    return pd.concat(data, ignore_index=True)
