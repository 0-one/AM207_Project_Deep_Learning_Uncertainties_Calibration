import numpy as np
import pandas as pd


def generate_data(func, points, seed=0):
    """Generate a dataframe containing the covariate X, and observations Y
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
