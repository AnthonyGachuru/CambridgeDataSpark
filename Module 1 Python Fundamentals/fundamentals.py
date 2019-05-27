# NUMPY exercises
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def zero_insert(x):
    '''
    Write a function that takes in a vector and returns a new vector where
    every element is separated by 4 consecutive zeros.

    Example:
    [4, 2, 1] --> [4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]

    :param x: input vector
    :type x:  numpy.array
    :returns: input vector with elements separated by 4 zeros
    :rtype:   numpy.array
    '''
    # 4 zeroes array
    myfourzeroes = [0, 0, 0, 0]

    # if the array is not 2 or more integers return the variable
    mylist = []
    if len(x) < 2:
        return x
    else:
        for element in x:
            mylist.append(element)
            mylist += myfourzeroes
    returnarray = np.array(mylist)
    return np.trim_zeros(returnarray)


def return_closest(x, val):
    '''
    Write a function that takes in a vector and returns the value contained in
    the vector that is closest to a given value.
    If two values are equidistant from val, return the one that comes first in
    the vector.

    Example:
    ([3, 4, 5], 2) --> 3

    :param x:   input vector
    :type x:    numpy.array of int/float
    :param val: input value
    :type val:  int/float
    :returns:   value from x closest to val
    :rtype:     int/float
    :raise:     ValueError
    '''
    if not any(x):
        return ValueError
    # set initial parameters
    differencenotset = True
    trackdifference = 0
    trackelement = 0

    # loop through elements in x
    for element in x:
        # calculate difference
        difference = element - val
        absdiff = np.abs(difference)
        # print("diff is {}".format(difference))
        # print("absdiff is {}".format(absdiff))
        if differencenotset:
            differencenotset = False
            trackdifference = absdiff
            trackelement = element
        else:
            if absdiff < trackdifference:
                trackdifference = absdiff
                trackelement = element
    return trackelement


def cauchy(x, y):
    '''
    Write a function that takes in two vectors
    and returns the associated Cauchy
    matrix with entries a_ij = 1/(x_i-y_j).

    Example:
    ([1, 2], [3, 4]) --> [[-1/2, -1/3], [-1, -1/2]]

    Note: the function should raise an error of type ValueError if there is a
    pair (i,j) such that x_i=y_j

    :param x: input vector
    :type x:  numpy.array of int/float
    :param y: input vector
    :type y:  numpy.array of int/float
    :returns: Cauchy matrix with entries 1/(x_i-y_j)
    :rtype:   numpy.array of float
    '''
    # while i < lengthofy and i > 2: need this because
    #     if y is only one long then you would only create one array
    if not any(x):
        return ValueError
    if not any(y):
        return ValueError
    filluparray = np.zeros(shape=(len(x), len(y)))
    # print(fillupArray)
    for xiterator in range(len(x)):
        x_i = x[xiterator]
        for yiterator in range(len(y)):
            y_j = y[yiterator]
            if x_i == y_j:
                return ValueError
            calcfield = 1 / (x_i - y_j)
            # print(calcfield)
            filluparray[xiterator, yiterator] = calcfield
    # print(fillupArray)
    return filluparray


def most_similar(x, v_list):
    '''
    Write a function that takes in a vector x and a list of vectors and finds,
    in the list, the index of the vector that is most similar to x in the
    cosine-similarity sense.

    Example:
    ([1, 1], [[1, 0.9], [-1, 1]]) --> 0 (corresponding to [1,0.9])

    :param x:      input vector
    :type x:       numpy.array of int/float
    :param v_list: list of vectors
    :type v_list:  list of numpy.array
    :returns:      index of element in list that is closest to x in cosine-sim
    :rtype:        int
    '''
    if not v_list:
        return 0
    # initial parameters
    entrynumber = 0
    c_sim_master = -1.1  # min value for c_sim is -1.

    for entry in range(len(v_list)):
        # cosine similarity
        numerator = np.dot(x, v_list[entry])
        nrm2 = np.sqrt(np.dot(x, x))
        nrm2_2 = np.sqrt(np.dot(v_list[entry], v_list[entry]))
        c_sim = numerator / (nrm2 * nrm2_2)

        # print(c_sim)
        if c_sim > c_sim_master:
            entrynumber = entry
            c_sim_master = c_sim

    return entrynumber


def gradient_descent(x_0, step, tol):
    '''
    Write a function that does a fixed-stepsize gradient descent on function f
    with gradient g and stops when the update has magnitude under a given
    tolerance level (i.e. when |xk-x(k-1)| < tol).
    Return a tuple with the position, the value of f at that position and the
    magnitude of the last update.
    h(x) = (x-1)^2 + exp(-x^2/2)
    f(x) = log(h(x))
    g(x) = (2(x-1) - x exp(-x^2/2)) / h(x)

    Example:
    (1.0, 0.1, 1e-3) --> approximately (1.2807, -0.6555, 0.0008)

    :param x_0:  initial point
    :type x_0:   float
    :param step: fixed step size
    :type step:  float
    :param tol:  tolerance for the magnitude of the update
    :type tol:   float
    :returns:    the position, the value at that position and the latest update
    :rtype:      tuple of three float
    '''
    # the lambdas here are equivalent to 'def f(x): return ... '
    h = lambda x: (x-1)**2 + np.exp(-x**2/2)
    g = lambda x: (2*(x-1) - x * np.exp(-x**2/2)) / h(x)
    f = lambda x: np.log(h(x))

    difference = tol+step
    xk_0 = x_0
    while difference > tol:
        # calculate the next value
        xkplus1 = xk_0 - (step * g(xk_0))
        # calculate the difference between the new place,
        # xkplus1, and the old place xk
        difference = xkplus1 - xk_0
        print(difference)
        # store the new place we have got to in xk
        xk_0 = xkplus1
    return (np.round(xk_0, 4), np.round(f(xk_0), 4), np.round(difference, 4))


def filter_rep(df):
    '''
    Write a function that takes a DataFrame with a colum `A` of integers and
    filters out the rows which contain the same value as a row above.
    Check that the index is right, use reset_index if necessary.

    Example:
        A   ...            A   ...
    ___________        ___________
    0 | 1 | ...        0 | 1 | ...
    1 | 1 | ...        1 | 0 | ...
    2 | 0 | ...  -->   2 | 5 | ...
    3 | 5 | ...        3 | 2 | ...
    4 | 5 | ...
    5 | 5 | ...
    6 | 2 | ...
    7 | 1 | ...

    :param df: input data frame with a column `A`
    :type df:  pandas.DataFrame
    :returns:  a dataframe where rows have been filtered out
    :rtype:    pandas.DataFrame
    '''
    if 'A' in df.columns:
        newdataframe = df.groupby('A', as_index=False, sort=False).first()
        newdataframe.reindex
        return newdataframe
    else:
        return df


def subtract_row_mean(df):
    '''
    Given a DataFrame of numeric values, write a function to subtract the row
    mean from each element in the row.

    Example:
        A   B   C                A     B     C
    _____________         _____________________
    0 | 1 | 5 | 0    -->  0 | -1.0 | 3.0 | -2.0
    1 | 2 | 6 | 1         1 | -1.0 | 3.0 | -2.0

    :param df: input data frame
    :type df:  pandas.DataFrame
    :returns:  a dataframe where each row is centred
    :rtype:    pandas.DataFrame
    '''
    demean = lambda x: x - x.mean()
    centreddf = df.transform(demean, axis=1)
    return centreddf


def all_unique_chars(string):
    '''
    Write a function to determine if a string is only made of unique
    characters and returns True if that's the case, False otherwise.
    Upper case and lower case should be considered as the same character.

    Example:
    "qwr#!" --> True, "q Qdf" --> False

    :param string: input string
    :type string:  string
    :returns:      true or false if string is made of unique characters
    :rtype:        bool
    '''
    lowerstring = string.lower()
    letterlist = []
    duplicatefound = False
    for letter in lowerstring:
        if letter in letterlist:
            duplicatefound = True
            # end for loop
            break
            #
        letterlist.append(letter)
        print(letter)
    return not duplicatefound


def find_element(sq_mat, val):
    '''
    Write a function that takes a square matrix of integers and returns the
    position (i,j) of a value. The position should be returned as a list of two
    integers. If the value is present multiple times, a single valid list
    should be returned.
    The matrix is structured in the following way:
    - each row has strictly decreasing values with the column index increasing
    - each column has strictly decreasing values with the row index increasing
    The following matrix is an example:

    Example:
    mat = [ [10, 7, 5],
            [ 9, 4, 2],
            [ 5, 2, 1] ]
    find_element(mat, 4) --> [1, 1]

    The function should raise an exception ValueError if the value isn't found.
    The time complexity of the function should be linear in the number of rows.

    :param sq_mat: the square input matrix with decreasing rows and columns
    :type sq_mat:  numpy.array of int
    :param val:    the value to be found in the matrix
    :type val:     int
    :returns:      the position of the value in the matrix
    :rtype:        list of int
    '''
    listofpoints = []
    for rownumber in range(len(sq_mat)):
        row = sq_mat[rownumber]
        # print("rownumber is {} and the row is {}".format(rownumber, row))
        for columnnumber in range(len(row)):
            column = row[columnnumber]
            if column == val:
                listofpoints.append(rownumber)
                listofpoints.append(columnnumber)
            if column < val:
                break
            # print("    columnnumber is {} and the value is
            # {}".format(columnnumber, column))
    if listofpoints == []:
        raise ValueError
    else:
        return listofpoints


def filter_matrix(mat):
    '''
    Write a function that takes an n x p matrix of integers and sets the rows
    and columns of every zero-entry to zero.

    Example:
    [ [1, 2, 3, 1],        [ [0, 2, 0, 1],
      [5, 2, 0, 2],   -->    [0, 0, 0, 0],
      [0, 1, 3, 3] ]         [0, 0, 0, 0] ]

    The complexity of the function should be linear in n and p.

    :param mat: input matrix
    :type mat:  numpy.array of int
    :returns:   a matrix where rows and columns of zero entries in mat are zero
    :retype:    numpy.array
    '''
    listofzerocolumns = []
    for rownumber in range(len(mat)):
        row = mat[rownumber]
        if 0 in row:
            # print("row {} has a zero".format(rownumber))
            # loop through elements in row and find where 0 is
            for item in range(len(row)):
                if row[item] == 0:
                    listofzerocolumns.append(item)
            mat[rownumber] = [0] * len(row)
    for entry in listofzerocolumns:
        mat.T[entry] = [0] * len(mat.T[entry])
    return mat


def largest_sum(intlist):
    '''
    Write a function that takes in a list of integers, finds the contiguous
    sublist with at least one element with the largest sum and returns the sum.
    If the list is empty, 0 should be returned.

    Example:
    [-1, 2, 2] --> 4 (corresponding to [2, 2])

    Time complexity target: linear in the number of integers in the list.

    :param intlist: input list of integers
    :type intlist:  list of int
    :returns:       the largest sum
    :rtype:         int
    '''
    # deal with the weird conditions
    if not intlist:
        return 0
    if len(intlist) == 1:
        return intlist[0]
    # we have a list of at least 2 weights
    runningsumset = False
    positivenumset = False
    negativenumset = False
    runningsum = 0
    highestvaluepos = 0
    highestvalueneg = 0
    for num, number in enumerate(intlist):
        print("num is {}".format(num))
        print("number is {}".format(number))
        if number < 0:
            if positivenumset:
                if runningsum > highestvaluepos:
                    highestvaluepos = runningsum
                runningsum = 0
            if not negativenumset:
                highestvalueneg = number
                negativenumset = True
                continue
            highestvalueneg = max(highestvalueneg, number)
            continue
        positivenumset = True
        runningsum += number
    # choose a value to return
    if runningsum > highestvaluepos:
        highestvaluepos = runningsum
    if positivenumset:
        return highestvaluepos
    else:
        return highestvalueneg


def pairprod(intlist, val):
    '''
    Write a function that takes in a list of positive integers (elements > 0)
    and returns all unique pairs of elements whose product is equal to a given
    value. The pairs should all be of the form (i, j) with i<=j.
    The ordering of the pairs does not matter.

    Example:
    ([3, 5, 1, 2, 3, 6], 6) --> [(2, 3), (1, 6)]

    Complexity target: subquadratic

    :param intlist: input list of integers
    :type intlist:  list of int
    :param val:     given value products will be compared to
    :type val:      int
    :returns:       pairs of elements such that the product of corresponding
                    entries matches the value val
    :rtype:         list of tuple
    '''
    listoflists = []
    for num, name in enumerate(intlist):
        if val % name != 0:
            continue
        # print("number {} is a factor".format(name))
        whileit = num + 1
        while whileit < len(intlist):
            secondnum = intlist[whileit]
            # print(secondnum)
            if (name * secondnum) == val:
                newlist = [name, secondnum]
                newlist.sort()
                if newlist not in listoflists:
                    listoflists.append(newlist)
            whileit += 1
    listoftuples = []
    for entry in listoflists:
        listoftuples.append(tuple(entry))
    return listoftuples


def draw_co2_plot():
    '''
    Here is some chemistry data

      Time (decade): 0, 1, 2, 3, 4, 5, 6
      CO2 concentration (ppm): 250, 265, 272, 260, 300, 320, 389

    Create a line graph of CO2 versus time, the line should be a blue dashed
    line. Add a title and axis titles to the plot.
    '''
    timedecade = [0, 1, 2, 3, 4, 5, 6]
    co2 = [250, 265, 272, 260, 300, 320, 389]

    plt.plot(timedecade, co2, 'b--')
    # labels
    plt.xlabel('Time (decade)')
    plt.ylabel('CO2 concentration (ppm)')
    # title
    plt.title('Chemistry data')
    plt.show()


def draw_equations_plot():
    '''
    Plot the following lines on the same plot

      y=cos(x) coloured in red with dashed lines
      y=x^2 coloured in blue with linewidth 3
      y=exp(-x^2) coloured in black

    Add a legend, title for the x-axis and a title to the curve, the x-axis
    should range from -4 to 4 and the y axis should range from 0 to 2. The
    figure should have a size of 8x6 inches.
    '''
    # create 100 values between -1 and 3
    x = np.linspace(-4, 4, 200)
    # compute the value of the function at those points
    cosfunc = np.cos(x)
    sqrfunc = x ** 2
    expfunc = np.exp(-x ** 2)
    plt.figure(figsize=(8, 6))
    plt.plot(x, cosfunc, ls='dashed', color='red', label='cos x')
    plt.plot(x, sqrfunc, color='blue', label='x squared')
    plt.plot(x, expfunc, color='black', label='exp -x squared')
    # axis limits
    plt.xlim(-4, 4)
    plt.ylim(0, 2)
    ##
    plt.xlabel("x label")
    ##
    plt.title("3 graphs on one sheet")
    ##
    plt.legend()
    ##
    plt.show()
