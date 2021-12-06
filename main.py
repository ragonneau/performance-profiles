#!/usr/bin/env python3
import os
import sys

os.environ.setdefault('ARCHDEFS', '/usr/local/opt/archdefs/libexec')
os.environ.setdefault('SIFDECODE', '/usr/local/opt/sifdecode/libexec')
os.environ.setdefault('MASTSIF', '/usr/local/opt/mastsif/share/mastsif')
os.environ.setdefault('CUTEST', '/usr/local/opt/cutest/libexec')
os.environ.setdefault('MYARCH', 'mac64.osx.gfo')
os.environ.setdefault('PYCUTEST_CACHE', 'archives')
sys.path.append(os.path.abspath(os.environ['PYCUTEST_CACHE']))
from perform import Profiles  # noqa


def validate(problem):
    valid = problem.m <= 50
    valid = valid and problem.name not in ['CSFI1', 'CSFI2', 'POLAK6']
    return valid


if __name__ == '__main__':
    p = Profiles(10, constraints='XB', callback=validate)
    p(solvers=['BOBYQA', 'COBYQA'])
