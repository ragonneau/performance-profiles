#!/usr/bin/env python3
import os
import sys
from pathlib import Path

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
os.environ.setdefault('PYCUTEST_CACHE', 'archives')

BASE_DIR = Path(__file__).resolve(strict=True).parent
ARCH_DIR = Path(BASE_DIR, os.environ.get('PYCUTEST_CACHE'))
ARCH_DIR.mkdir(exist_ok=True)
from perform import Profiles  # noqa


def validate(problem):
    valid = problem.m <= 1000
    return valid


if __name__ == '__main__':
    profiles = Profiles(50, constraints='QO', callback=validate)
    profiles(['cobyqa', 'cobyla', 'slsqp'], load=True)
