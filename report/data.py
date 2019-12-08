import numpy
import pandas as pd


def generate_data(func, points, seed=0):
    """Generate a dataframe containing the covariate X, and observations Y
    """
    numpy.random.seed(seed)

    data = []
    for segment in points:
        x = numpy.linspace(*segment["xlim"], num=segment["n_points"])
        distribution = func(x)
        # Generate observations
        y = distribution.rvs()
        df = pd.DataFrame({"x": x, "y": y})
        data.append(df)

    return pd.concat(data, ignore_index=True)
