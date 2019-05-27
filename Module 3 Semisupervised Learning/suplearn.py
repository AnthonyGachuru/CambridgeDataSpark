import re
from datetime import datetime
import numpy as np


def diff_in_days(first_list, second_list):
    """
    Write a function that takes two lists of UNIX timestamps as input and
    returns a list consisting of the absolute difference in days between
    the two lists.

    Example input:
    first_list = [1472688000, 1477958400] # corresponds to 01/09/16 , 01/11/16
    second_list = [1488326400, 1491004800] # corresponds to 01/03/17 , 01/04/17

    Output diff_in_days(first_list, second_list) --> [181, 152]

    Note:
    https://en.wikipedia.org/wiki/Unix_time,
    https://docs.python.org/2/library/datetime.html

    :param first_list: list of unix timestamps
    :param second_list: list of unix timestamps
    of the same length as first_list
    :return: list of absolute difference in days
    :rtype: list
    """
    listtoholddiffs = []
    for firstdate, seconddate in zip(first_list, second_list):
        datetime1 = datetime.fromtimestamp(firstdate)
        datetime2 = datetime.fromtimestamp(seconddate)
        diff = datetime1 - datetime2
        listtoholddiffs.append(np.abs(diff.days))

    return listtoholddiffs


def find_2nd(string, substring):
    return string.find(substring, string.find(substring) + 1)


def return_location(list_locations):
    """
    Write a function that takes a list of strings each containing information
    about a specific location and returns a list of locations. The locations
    are always preceded by "short_name:" and always in the format
    "City Name, XX" where XX is a two-letter indicator of the US state.

    Note: you can assume all strings are exactly in the
    format given below though
    possibly longer and with different keys.

    Example input:
    string1 =
      "{""displayable_name"":""Detroit, MI"",
      ""short_name"":""Detroit, MI"",""id"":2391585,""state"":""MI""}"
    string2 =
      "{""displayable_name"":""Tracy, CA"",
      ""short_name"":""Tracy, CA"",""id"":2507550,""state"":""CA""}"
    strlist = [string1, string2]

    Output of return_location(strlist): ["Detroit, MI", "Tracy, CA"]

    :param list_locations: the list of strings
    :return: list of locations identified by "short_name"
    :rtype: list
    """
    cityandstate = []
    for cityinfo in list_locations:
        nospeechmarks = cityinfo.replace('"', '')
        startofword = nospeechmarks.find('short_name:')
        endofword = find_2nd(nospeechmarks[startofword:], ',')
        cityandstate.append(
            nospeechmarks[startofword + 11:startofword + endofword])
    return cityandstate


def return_post_codes(text):
    """
    Write a function that takes an arbitrary text and returns a list of
    post-codes that appear in that text. Postcodes, in the UK, are of one of
    the following form where `X` means a letter appears and `9` means a number
    appears:

    X9 9XX
    X9X 9XX
    X99 9XX
    XX9 9XX
    XX9X 9XX
    XX99 9XX

    Note that even though the standard layout is to include one single space
    in between the two halves of the post code, there are occasional formating
    errors where an arbitrary number of space is included (0, 1, or more). You
    should parse those codes as well.

    :param text: a raw, arbitrary text
    :return: list of post-codes
    :rtype: list
    """
    valid = re.compile(r"[a-zA-Z]{1,2}[0-9]{1,2}[a-zA-Z]?" +
                       "[ ]*[0-9]{1}[a-zA-Z]{2}")
    allmatches = valid.findall(text)
    formattedmatches = []
    for postcode in allmatches:
        formattedmatches.append(re.sub(r'\s+', ' ', postcode).strip())
    return formattedmatches
