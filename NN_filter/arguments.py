# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:43:13 2020

@author: adamw
"""

import argparse

parser = argparse.ArgumentParser()

parser.parse_args(
        '--name', required=True, type=str,
        help='Name of the output file'
        )

parser.parse_args(
        '--nruns', default=25, type=int,
        help='Number of times the network trains'
        )

parser.parse_args(
        '--num_epochs', default=50, type=int,
        help='Number of epochs that are run'
        )

parser.parse_args(
        '--neurons', default=64, type=int,
        help='Number of neurons in each layer'
        )

parser.parse_args(
        '--encoder', default=0.3, type=float,
        help='How to encode the data'
        )

parser.parse_args(
        '--dropout', default=0.3, type=float,
        help='Dropout'
        )

parser.parse_args(
        '--lr', default=0.0001, type=float,
        help='Learning rate'
        )