# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 02:27:14 2023

@author: David J. Kedziora
"""

# import autonoml as aml
from autonoml.data import (DataFormatX, DataFormatY, 
                           reformat_x, reformat_y)

x = {"f_1":["a","b","c"], "f_2":[1,2,3]}
x_keys = x.keys()
y = [True, False, True]

def test_conversions_x(in_x, in_format_old):
    for format_new in DataFormatX:
        if format_new.value >= in_format_old.value:
            print("%s -> %s" % (in_format_old.name, format_new.name))
            x_forward = reformat_x(in_data = in_x,
                                   in_format_old = in_format_old,
                                   in_format_new = format_new,
                                   in_keys_features = x_keys)
            print(in_x)
            print("...to...")
            print(x_forward)
            print()

            if not format_new == in_format_old:
                print("%s -> %s" % (format_new.name, in_format_old.name))
                x_back = reformat_x(in_data = x_forward,
                                    in_format_old = format_new,
                                    in_format_new = in_format_old,
                                    in_keys_features = x_keys)
                print(x_forward)
                print("...to...")
                print(x_back)
                print()

                test_conversions_x(in_x = x_forward, in_format_old = format_new)

def test_conversions_y(in_y, in_format_old):
    for format_new in DataFormatY:
        if format_new.value >= in_format_old.value:
            print("%s -> %s" % (in_format_old.name, format_new.name))
            y_forward = reformat_y(in_data = in_y,
                                   in_format_old = in_format_old,
                                   in_format_new = format_new)
            print(in_y)
            print("...to...")
            print(y_forward)
            print()

            if not format_new == in_format_old:
                print("%s -> %s" % (format_new.name, in_format_old.name))
                y_back = reformat_y(in_data = y_forward,
                                    in_format_old = format_new,
                                    in_format_new = in_format_old)
                print(y_forward)
                print("...to...")
                print(y_back)
                print()

                test_conversions_y(in_y = y_forward, in_format_old = format_new)

test_conversions_x(in_x = x, in_format_old = list(DataFormatX)[0])
test_conversions_y(in_y = y, in_format_old = list(DataFormatY)[0])