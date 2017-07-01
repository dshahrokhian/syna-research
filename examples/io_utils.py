# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Input/Output utils
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

import csv

def create_csv(filename, fields):
    with open(filename,'wb') as f:
        f.write(fields, delimiter=',')

def append_csv(filename, results):
    with open(filename, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(results)