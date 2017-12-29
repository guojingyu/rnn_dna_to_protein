"""
Utility functions

Author Jingyu Guo
"""

import datetime

def today():
    """
    :return: a string representing today
    """
    return datetime.datetime.now().strftime("%Y-%m-%d")

def clock_now():
    """
    :return: return a formatted string as current time of the day
    """
    return datetime.datetime.now().strftime("%H:%M:%S")

def time_stamp():
    """
    :return: return a time stamp string as current time of the day
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")