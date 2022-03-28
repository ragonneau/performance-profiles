#!/usr/bin/env python3
import os
import sys

import numpy as np

np.set_printoptions(
    precision=4,
    linewidth=sys.maxsize,
    sign=' ',
)

if sys.platform == 'linux':
    os.environ.setdefault('ARCHDEFS', '/opt/cutest/archdefs')
    os.environ.setdefault('SIFDECODE', '/opt/cutest/sifdecode')
    os.environ.setdefault('CUTEST', '/opt/cutest/cutest')
    os.environ.setdefault('MASTSIF', '/opt/cutest/mastsif')
    os.environ.setdefault('MYARCH', 'pc64.lnx.gfo')
elif sys.platform == 'darwin':
    os.environ.setdefault('ARCHDEFS', '/usr/local/opt/archdefs/libexec')
    os.environ.setdefault('SIFDECODE', '/usr/local/opt/sifdecode/libexec')
    os.environ.setdefault('CUTEST', '/usr/local/opt/cutest/libexec')
    os.environ.setdefault('MASTSIF', '/usr/local/opt/mastsif/share/mastsif')
    os.environ.setdefault('MYARCH', 'mac64.osx.gfo')
else:
    raise NotImplementedError
sys.path.append(f'{os.environ["HOME"]}/PycharmProjects/cobyqa')
from perfprof import Profiles  # noqa


def validate(problem):
    valid = problem.m <= 1000
    return valid


if __name__ == '__main__':
    profiles = Profiles(10, constraints='U', callback=validate)
    profiles(['cobyqa', 'newuoa'], load=False)
    profiles(['cobyqa', 'bobyqa'], load=False)
    profiles(['cobyqa', 'lincoa'], load=False)
    profiles(['cobyqa', 'cobyla'], load=False)
    del profiles

    profiles = Profiles(10, constraints='B', callback=validate)
    profiles(['cobyqa', 'bobyqa'], load=False)
    profiles(['cobyqa', 'lincoa'], load=False)
    profiles(['cobyqa', 'cobyla'], load=False)
    del profiles

    profiles = Profiles(10, constraints='NL', callback=validate)
    profiles(['cobyqa', 'lincoa'], load=False)
    profiles(['cobyqa', 'cobyla'], load=False)
    del profiles

    profiles = Profiles(10, constraints='QO', callback=validate)
    profiles(['cobyqa', 'cobyla'], load=False)
    del profiles
