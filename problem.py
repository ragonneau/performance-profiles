import re
import warnings
from subprocess import DEVNULL, PIPE, Popen

import numpy as np
import pycutest
from scipy.optimize import Bounds, minimize


class CUTEstProblems(list):

    def __init__(self, n, constraints, callback=None):
        super().__init__()
        self._n = n
        names = pycutest.find_problems('CLQSO', constraints, n=[1, self._n])
        for i, name in enumerate(sorted(names)):
            try:
                print(f'Loading {name} ({i + 1}/{len(names)})...', end=' ')
                if pycutest.problem_properties(name)['n'] is None:
                    sif = self._sif_decode(name)
                    if sif.size > 0 and sif[0] <= self._n:
                        params = {'N': np.max(sif[sif <= self._n])}
                        problem = CUTEstProblem(name, sifParams=params)
                        self.append(problem, callback)
                    else:
                        print('No compliant SIF parameters found.')
                else:
                    problem = CUTEstProblem(name)
                    self.append(problem, callback)
            except (AttributeError, ModuleNotFoundError, RuntimeError):
                print('Internal errors occurred.')

    def append(self, p, callback=None):
        if self._validate(p, callback):
            super().append(p)
            print('Loading successful.')
        else:
            print('Validation failed.')

    def _validate(self, problem, callback=None):
        valid = isinstance(problem, CUTEstProblem)
        valid = valid and np.all(problem.vartype == 0)
        valid = valid and problem.n <= self._n
        if callback is not None:
            valid = valid and callback(problem)
        return valid

    @staticmethod
    def _sif_decode(name, param='N'):
        cmd = [pycutest.get_sifdecoder_path(), '-show', name]
        sp = Popen(cmd, universal_newlines=True, stdout=PIPE, stderr=DEVNULL)
        sif_stdout = sp.stdout.read()
        sp.wait()

        regex = re.compile(r'^(?P<param>[A-Z]+)=(?P<value>\d+)')
        sif = []
        for stdout in sif_stdout.split('\n'):
            sif_match = regex.match(stdout)
            if sif_match and sif_match.group('param') == param:
                sif.append(int(sif_match.group('value')))
        return np.sort(sif)


class CUTEstProblem:

    def __init__(self, *args, **kwargs):
        self._p = pycutest.import_problem(*args, **kwargs)
        self._xl = None
        self._xu = None
        self._aub = None
        self._bub = None
        self._aeq = None
        self._beq = None
        self._project_initial_guess()

    def __getattr__(self, item):
        try:
            return getattr(self._p, item)
        except AttributeError as exc:
            raise AttributeError(item) from exc

    def __str__(self):
        return str(self._p)

    @property
    def xl(self):
        if self._xl is not None:
            return self._xl
        self._xl = self.bl
        self._xl[self._xl <= -1e20] = -np.inf
        return self._xl

    @property
    def xu(self):
        if self._xu is not None:
            return self._xu
        self._xu = self.bu
        self._xu[self._xu >= 1e20] = np.inf
        return self._xu

    @property
    def aub(self):
        if self._aub is not None:
            return self._aub
        self._aub, self._bub = self._linear_ub()
        return self._aub

    @property
    def bub(self):
        if self._bub is not None:
            return self._bub
        self._aub, self._bub = self._linear_ub()
        return self._bub

    @property
    def aeq(self):
        if self._aeq is not None:
            return self._aeq
        self._aeq, self._beq = self._linear_eq()
        return self._aeq

    @property
    def beq(self):
        if self._beq is not None:
            return self._beq
        self._aeq, self._beq = self._linear_eq()
        return self._beq

    @property
    def mlub(self):
        if self.m == 0:
            return 0
        iub = self.is_linear_cons & np.logical_not(self.is_eq_cons)
        iub_lower = self.cl[iub] > -1e20
        iub_upper = self.cu[iub] < 1e20
        return np.count_nonzero(iub_lower) + np.count_nonzero(iub_upper)

    @property
    def mleq(self):
        if self.m == 0:
            return 0
        ieq = self.is_linear_cons & self.is_eq_cons
        return np.count_nonzero(ieq)

    @property
    def mnlub(self):
        if self.m == 0:
            return 0
        iub = np.logical_not(self.is_linear_cons | self.is_eq_cons)
        iub_lower = self.cl[iub] > -1e20
        iub_upper = self.cu[iub] < 1e20
        return np.count_nonzero(iub_lower) + np.count_nonzero(iub_upper)

    @property
    def mnleq(self):
        if self.m == 0:
            return 0
        ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
        return np.count_nonzero(ieq)

    @property
    def type(self):
        properties = pycutest.problem_properties(self.name)
        return properties.get('constraints')

    def fun(self, x, gradient=False):
        x = np.asarray(x)
        return self.obj(x, gradient)

    def hess(self, x):
        x = np.asarray(x)
        return self.hess(x)

    def cub(self, x):
        if self.m == 0:
            return np.empty(0)
        x = np.asarray(x)
        iub = np.logical_not(self.is_linear_cons | self.is_eq_cons)
        iub_lower = self.cl[iub] > -1e20
        iub_upper = self.cu[iub] < 1e20
        cx = []
        for i, index in enumerate(np.flatnonzero(iub)):
            c_index = self.cons(x, index)
            if iub_lower[i]:
                cx.append(self.cl[index] - c_index)
            if iub_upper[i]:
                cx.append(c_index - self.cu[index])
        return np.array(cx)

    def cub_jac(self, x):
        if self.m == 0:
            return np.empty((0, self.n))
        x = np.asarray(x)
        iub = np.logical_not(self.is_linear_cons | self.is_eq_cons)
        iub_lower = self.cl[iub] > -1e20
        iub_upper = self.cu[iub] < 1e20
        gx = []
        for i, index in enumerate(np.flatnonzero(iub)):
            _, g_index = self.cons(x, index, True)
            if iub_lower[i]:
                gx.append(-g_index)
            if iub_upper[i]:
                gx.append(g_index)
        return np.reshape(gx, (-1, self.n))

    def ceq(self, x):
        if self.m == 0:
            return np.empty(0)
        x = np.asarray(x)
        ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
        cx = []
        for index in np.flatnonzero(ieq):
            c_index = self.cons(x, index)
            cx.append(c_index - 0.5 * (self.cl[index] + self.cu[index]))
        return np.array(cx)

    def ceq_jac(self, x):
        if self.m == 0:
            return np.empty((0, self.n))
        x = np.asarray(x)
        ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
        gx = []
        for index in np.flatnonzero(ieq):
            _, g_index = self.cons(x, index, True)
            gx.append(g_index)
        return np.reshape(gx, (-1, self.n))

    def maxcv(self, x):
        violmx = np.max(self.xl - x, initial=0.0)
        violmx = np.max(x - self.xu, initial=violmx)
        violmx = np.max(np.dot(self.aub, x) - self.bub, initial=violmx)
        violmx = np.max(np.abs(np.dot(self.aeq, x) - self.beq), initial=violmx)
        violmx = np.max(self.cub(x), initial=violmx)
        violmx = np.max(np.abs(self.ceq(x)), initial=violmx)
        return violmx

    def _linear_ub(self):
        if self.m == 0:
            return np.empty((0, self.n)), np.empty(0)
        iub = self.is_linear_cons & np.logical_not(self.is_eq_cons)
        iub_lower = self.cl[iub] > -1e20
        iub_upper = self.cu[iub] < 1e20
        aub = []
        bub = []
        for i, index in enumerate(np.flatnonzero(iub)):
            c_index, g_index = self.cons(np.zeros(self.n), index, True)
            if iub_lower[i]:
                aub.append(-g_index)
                bub.append(c_index - self.cl[index])
            if iub_upper[i]:
                aub.append(g_index)
                bub.append(self.cu[index] - c_index)
        return np.reshape(aub, (-1, self.n)), np.array(bub)

    def _linear_eq(self):
        if self.m == 0:
            return np.empty((0, self.n)), np.empty(0)
        ieq = self.is_linear_cons & self.is_eq_cons
        aeq = []
        beq = []
        for index in np.flatnonzero(ieq):
            c_index, g_index = self.cons(np.zeros(self.n), index, True)
            aeq.append(g_index)
            beq.append(0.5 * (self.cl[index] + self.cu[index]) - c_index)
        return np.reshape(aeq, (-1, self.n)), np.array(beq)

    def _project_initial_guess(self):
        bounds = Bounds(self.xl, self.xu)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(self._cpqp_obj, self.x0, jac=True, bounds=bounds)
        self.x0 = res.x

    def _cpqp_obj(self, x):
        cub = np.maximum(0.0, np.dot(self.aub, x) - self.bub)
        ceq = np.dot(self.aeq, x) - self.beq
        fx = 0.5 * (np.inner(cub, cub) + np.inner(ceq, ceq))
        gx = np.dot(self.aub.T, cub) + np.dot(self.aeq.T, ceq)
        return fx, gx
