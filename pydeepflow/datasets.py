import pandas as pd
from sklearn.utils import Bunch 


def load_dataset(filename,descr="PyDeepFlow Datatset"):
    """
    
    Load dataset from CSV file.

    Parameters
    ----------
    filename (str): Path to the CSV file.
    descr (str): Description of the dataset.

    Returns
    ----------
    Bunch: A Bunch object containing the dataset information.
    
    """
    df = pd.read_csv(filename)

    x = df.iloc[:,:-1]
    y=df.iloc[:,-1]

    return Bunch(
        data=x.values,
        target=y.values,
        frame=df,
        target_names = y.unique().tolist(),
        feature_names = x.columns.tolist(),
        DESCR=descr,

    )


def load_iris(return_X_y=False,as_frame=False):
    """
    Load the Iris dataset.

    Parameters
    ----------
    return_X_y (bool): If True, returns (data, target) instead of a Bunch object.
    as_frame (bool): If True, returns a pandas DataFrame instead of a Bunch object.

    Returns
    ----------
    Bunch or (data, target) or DataFrame.
    
    """

    bunch = load_dataset("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",descr="Iris Dataset")

    if return_X_y:
        return bunch.data,bunch.target
    if as_frame:
        return bunch.frame
    return bunch


def load_wine(return_X_y=False,as_frame=False):
    """
    Load the Wine Quality dataset.

    Parameters
    ----------
    return_X_y (bool): If True, returns (data, target) instead of a Bunch object.
    as_frame (bool): If True, returns a pandas DataFrame instead of a Bunch object.

    Returns
    ----------
    Bunch or (data, target) or DataFrame.
    """

    bunch = load_dataset("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",descr="Wine Quality Dataset")

    if return_X_y:
        return bunch.data,bunch.target
    if as_frame:
        return bunch.frame
    return bunch


def load_digits(return_X_y=False,as_frame=False):
    """
    Load the Optical Recognition of Handwritten Digits dataset.

    Parameters
    ----------
    return_X_y (bool): If True, returns (data, target) instead of a Bunch object.
    as_frame (bool): If True, returns a pandas DataFrame instead of a Bunch

    Returns
    ----------
    Bunch or (data, target) or DataFrame.
    """

    bunch = load_dataset("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",descr="Optical Recognition of Handwritten Digits Dataset")

    if return_X_y:
        return bunch.data,bunch.target
    if as_frame:
        return bunch.frame
    return bunch