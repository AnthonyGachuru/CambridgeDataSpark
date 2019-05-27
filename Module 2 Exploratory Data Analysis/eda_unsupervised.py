import numpy as np
import pandas as pd


def nan_processor(df, replacement_str=''):
    """
    Take a DataFrame and return one where all occurrences
    of the replacement string have been replaced by `np.nan`
    and, consequently, all rows containing np.nan
    have been removed.

    Example with replacement_str='blah'
         A       B      C                   A     B    C
    --------------------------         ------------------
    0 |  0.5 |  0.3   | 'blah'         1 | 0.2 | 0.1 | 5
    1 |  0.2 |  0.1   |   5     -->    3 | 0.7 | 0.2 | 1
    2 |  0.1 | 'blah' |   3
    3 |  0.7 |  0.2   |   1

    :param df: Input data frame (pandas.DataFrame)
    :param replacement_str: string to find and replace by np.nan
    :returns: DataFrame where the occurences of replacement_str have been
        replaced by np.nan and subsequently all rows containing np.nan have
        been removed
    :rtype: pandas.DataFrame
    """
    df = df.replace(to_replace=replacement_str, value=np.nan).dropna()
    return df


def feature_cleaner(df, low=0.05, high=0.95):
    """
    Take a dataframe where columns are all numerical and non-constant.
    For each feature, mark the values that are not between the given
    percentiles (low-high) as np.nan.
    Then, remove all rows containing np.nan.
    Finally, the columns must be scaled to have zero mean and unit variance
    (do this without sklearn).

    Example testdf:
            0     1     2
    ---------------------
    A |   0.1   0.2   0.1
    B |   5.0  10.0  20.0
    C |   0.2   0.3   0.5
    D |   0.3   0.2   0.7
    E |  -0.1  -0.2  -0.4
    F |   0.1   0.4   0.3
    G |  -0.5   0.3  -0.2
    H | -10.0   0.3   1.0

    Output of feature_cleaner(testdf, 0.01, 0.99):

                0         1         2
    ---------------------------------
    A |  0.191663 -0.956183 -0.515339
    C |  0.511101  0.239046  0.629858
    D |  0.830540 -0.956183  1.202457
    F |  0.191663  1.434274  0.057260
    G | -1.724967  0.239046 -1.374236

    :param df:      Input DataFrame (with numerical columns)
    :param low:     Lowest percentile  (0.0<low<1.0)
    :param high:    Highest percentile (low<high<1.0)
        :returns:      Scaled DataFrame where elements that are outside of the
                    desired percentiel range have been removed
    :rtype: pandas.DataFrame
    """
    # For each feature, mark the values that are not between the given
    # percentiles (low-high) as np.nan.
    aboves = df > df.quantile(low)
    belows = df < df.quantile(high)

    df = df[aboves & belows]

    # Then, remove all rows containing np.nan.
    df.dropna(inplace=True)

    # Finally, the columns must be scaled to have zero mean and unit variance
    # this means deducting the mean and dividing by the standard deviation
    # df = ((df - df.mean(axis=1)) / df.std(axis=1))
    # df = df.sub(df.mean(axis=1),axis=1)
    # return dataframe
    # print(df)
    # print("Mean is {}".format(df.mean(axis=0)))
    df = df.subtract(df.mean(axis=0))
    # print(df)
    # print("std is {}".format(df.std(axis=0)))
    df = df.divide(df.std(axis=0))
    return df


def get_feature(df):
    """
    Take a dataframe where all columns are numerical and not constant.
    One of the column named "CLASS" is either 0 or 1.
    Within each class, for each feature compute the ratio (R) of the
    range over the variance (the range is the gap between the smallest
    and largest value).
    For each feature you now have two R (R_0 and R_1).
    For each feature, compute the ratio
    (say K) of the larger R to the smaller R.
    Return the name of the feature for which this last ratio K is largest.

    Test input
           A     B     C   CLASS
    ----------------------------
    0 |  0.1   0.2   0.1     0
    1 |  5.0  10.0  20.0     0
    2 |  0.2   0.3   0.5     1
    3 |  0.3   0.2   0.7     0
    4 |	-0.1  -0.2  -0.4     1
    5 |	 0.1   0.4   0.3     0
    6 |	-0.5   0.3  -0.2     0

    Output of get_feature(testdf) is 'C'

    :param df:  Input DataFrame (with numerical columns)
    :returns:   Name of the feature
    :rtype: str
    """
    # old code which is wrong
    # because we wanted the
    # ratio of range / variance for the feature in each class
    # class0 = df[df['CLASS'] == 0]
    # var0 = class0.var(axis=0)
    # class1 = df[df['CLASS'] == 1]
    # var1 = class1.var(axis=0)
    # diffs = var1 - var0
    # diffs.drop('CLASS', inplace=True)
    # absolutediffs = diffs.abs()
    # return absolutediffs.idxmax()
    class0 = df[df['CLASS'] == 0]
    var0 = class0.var(axis=0)
    range0 = class0.max(axis=0) - class0.min(axis=0)
    range_over_var = range0 / var0

    class1 = df[df['CLASS'] == 1]
    var1 = class1.var(axis=0)
    range1 = class1.max(axis=0) - class1.min(axis=0)
    range1_over_var = range1 / var1

    # create a dataframe of the ratios and compute the ratio of those
    newdf = pd.DataFrame(data=range_over_var, columns=['class0'])
    newdf = newdf.merge(
        range1_over_var.to_frame(), left_index=True, right_index=True)
    newdf['ratio'] = newdf.apply(
        lambda row: np.max(row) / np.min(row), axis=1)

    return newdf['ratio'].idxmax()


def one_hot_encode(label_to_encode, labels):
    """
    Write a function that takes in a label to encode and a list of possible
    labels. It should return the label one-hot-encoded as a list of elements
    containing 0s and a unique 1 at the index corresponding to the matching
    label. Note that the input list of labels should contain unique elements.
    This function should raise a ValueError if the label_to_encode
    can not be found in list of labels.

    Examples:
    one_hot_encode("blue", ["blue", "red", "pink", "yellow"]) -> [1, 0, 0, 0]
    one_hot_encode("pink", ["blue", "red", "pink", "yellow"]) -> [0, 0, 1, 0]
    one_hot_encode("b", ["a", "b", "c", "d", "e"]) -> [0, 1, 0, 0, 0]

    :param label_to_encode: the label to encode
    :param labels: a list of all possible labels
    :return: a list of 0s and one 1
    :rtype: pandas.DataFrame
    :raise ValueError:
    """
    # This function should raise a ValueError if the label_to_encode
    # can not be found in list of labels.
    if not labels:
        return ValueError
    if label_to_encode not in labels:
        return ValueError
    # Note that the input list of labels should contain unique elements
    if np.unique(labels).size != len(labels):
        return ValueError

    onehotencoding = [0] * len(labels)
    onehotencoding[labels.index(label_to_encode)] = 1
    return onehotencoding
