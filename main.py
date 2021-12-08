#!/usr/bin/env python3
import os
import sys
from pathlib import Path

if sys.platform == 'linux':
    os.environ.setdefault('ARCHDEFS', '/opt/cutest/archdefs')
    os.environ.setdefault('SIFDECODE', '/opt/cutest/sifdecode')
    os.environ.setdefault('MASTSIF', '/opt/cutest/mastsif')
    os.environ.setdefault('CUTEST', '/opt/cutest/cutest')
    os.environ.setdefault('MYARCH', 'pc64.lnx.gfo')
elif sys.platform == 'darwin':
    os.environ.setdefault('ARCHDEFS', '/usr/local/opt/archdefs/libexec')
    os.environ.setdefault('SIFDECODE', '/usr/local/opt/sifdecode/libexec')
    os.environ.setdefault('MASTSIF', '/usr/local/opt/mastsif/share/mastsif')
    os.environ.setdefault('CUTEST', '/usr/local/opt/cutest/libexec')
    os.environ.setdefault('MYARCH', 'mac64.osx.gfo')
else:
    raise NotImplementedError
os.environ.setdefault('PYCUTEST_CACHE', 'archives')
sys.path.append(os.path.abspath(os.environ['PYCUTEST_CACHE']))
BASE_DIR = Path(__file__).resolve(strict=True).parent
ARCH_DIR = Path(BASE_DIR, os.environ.get('PYCUTEST_CACHE'))
ARCH_DIR.mkdir(exist_ok=True)
from perform import Profiles  # noqa


def validate(problem):
    valid = problem.m <= 50
    valid = valid and problem.name not in ['CSFI1', 'CSFI2', 'POLAK6']
    # TODO: Study and remove the following
    valid = valid and problem.name not in ['HS84']
    return valid


if __name__ == '__main__':
    p = Profiles(10, constraints='U', callback=validate)
    p(solvers=['cobyqa', 'newuoa'])
