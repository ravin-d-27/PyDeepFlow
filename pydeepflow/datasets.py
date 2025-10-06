import pandas as pd
from sklearn.utils import Bunch 

def load_dataset(filename,descr="PyDeepFlow Datatset"):
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
    bunch = load_dataset("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",descr="Iris Dataset")

    if return_X_y:
        return bunch.data,bunch.target
    if as_frame:
        return bunch.frame
    return bunch


def load_wine(return_X_y=False,as_frame=False):
    bunch = load_dataset("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",descr="Wine Quality Dataset")

    if return_X_y:
        return bunch.data,bunch.target
    if as_frame:
        return bunch.frame
    return bunch

def load_digits(return_X_y=False,as_frame=False):
    bunch = load_dataset("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",descr="Optical Recognition of Handwritten Digits Dataset")

    if return_X_y:
        return bunch.data,bunch.target
    if as_frame:
        return bunch.frame
    return bunch