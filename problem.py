import re
import warnings
from subprocess import DEVNULL, PIPE, Popen

import numpy as np
import pycutest
from joblib import Parallel, delayed
from scipy.linalg import lstsq
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize


class CUTEstProblems(list):

    def __init__(self, n_min, n_max, constraints, callback=None):
        super().__init__()
        self._n_min = n_min
        self._n_max = n_max
        names = pycutest.find_problems('CLQSO', constraints, True, n=[self._n_min, self._n_max], userM=False)
        attempts = Parallel(n_jobs=-1)(self._load(sorted(names), i) for i in range(len(names)))
        for problem in attempts:
            if problem is not None:
                self.append(problem, callback)

    def append(self, problem, callback=None):
        if self._validate(problem, callback):
            super().append(problem)
        else:
            print(f'{problem.name}: validation failed.')

    @delayed
    def _load(self, names, i):
        try:
            print(f'Attempt loading {names[i]} ({i + 1}/{len(names)}).')
            if pycutest.problem_properties(names[i])['n'] is None:
                if names[i] in ['ARGLALE', 'ARGLBLE', 'ARGLCLE', 'BA-L16LS',
                                'BA-L21', 'BA-L21LS', 'BA-L49', 'BA-L49LS',
                                'BA-L52', 'BA-L52LS', 'BA-L73', 'BA-L73LS',
                                'BDRY2', 'CHANDHEU', 'CHARDIS0', 'CHARDIS1',
                                'DANWOODLS', 'DMN15102', 'DMN15102LS',
                                'DMN15103', 'DMN15103LS', 'DMN15332',
                                'DMN15332LS', 'DMN15333', 'DMN15333LS',
                                'DMN37142', 'DMN37142LS', 'DMN37143',
                                'DMN37143LS', 'GAUSS1LS', 'GAUSS2LS',
                                'GAUSS3LS', 'GAUSSELM', 'GOFFIN', 'GPP',
                                'KOEBHELB', 'LEUVEN3', 'LEUVEN4', 'LEUVEN5',
                                'LEUVEN6', 'LHAIFAM', 'LINCONT', 'LIPPERT2',
                                'LOBSTERZ', 'MGH17LS', 'MISRA1ALS', 'MISRA1CLS',
                                'MODEL', 'NASH', 'NELSONLS', 'OSBORNEA', 'PDE1',
                                'PDE2', 'PENALTY3', 'RAT43LS', 'RDW2D51F',
                                'RDW2D51U', 'RDW2D52B', 'RDW2D52F', 'RDW2D52U',
                                'ROSEPETAL', 'TWOD', 'WALL100', 'YATP1SQ',
                                'YATP2SQ']:
                    print(f'{names[i]}: no compilation attempted.')
                else:
                    sif = self._sif_decode(names[i])
                    mask = (sif >= self._n_min) & (sif <= self._n_max)
                    if np.any(mask):
                        return CUTEstProblem(names[i], sifParams={'N': np.max(sif[mask])}, drop_fixed_variables=False)
                    else:
                        print(f'{names[i]}: no compliant SIF parameters found.')
            else:
                problem = CUTEstProblem(names[i], drop_fixed_variables=False)
                return problem
        except (AttributeError, ModuleNotFoundError, RuntimeError, FileNotFoundError):
            print(f'{names[i]}: internal errors occurred.')

    def _validate(self, problem, callback=None):
        valid = isinstance(problem, CUTEstProblem)
        valid = valid and np.all(problem.vartype == 0)
        valid = valid and problem.n >= self._n_min
        valid = valid and problem.n <= self._n_max
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
        self._project_x0()

    def __getattr__(self, item):
        try:
            return getattr(self._p, item)
        except AttributeError as exc:
            raise AttributeError(item) from exc

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return str(self._p)

    @property
    def xl(self):
        if self._xl is not None:
            return self._xl
        self._xl = np.array(self.bl, dtype=float)
        self._xl[self._xl <= -1e20] = -np.inf
        return self._xl

    @property
    def xu(self):
        if self._xu is not None:
            return self._xu
        self._xu = np.array(self.bu, dtype=float)
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
        x = np.asarray(x, dtype=float)
        return self.obj(x, gradient)

    def hess(self, x):
        x = np.asarray(x, dtype=float)
        return self.hess(x)

    def cub(self, x):
        if self.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
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
        return np.array(cx, dtype=float)

    def cub_jac(self, x):
        if self.m == 0:
            return np.empty((0, self.n))
        x = np.asarray(x, dtype=float)
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
        gx = np.array(gx, dtype=float)
        return np.reshape(gx, (-1, self.n))

    def ceq(self, x):
        if self.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
        ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
        cx = []
        for index in np.flatnonzero(ieq):
            c_index = self.cons(x, index)
            cx.append(c_index - 0.5 * (self.cl[index] + self.cu[index]))
        return np.array(cx, dtype=float)

    def ceq_jac(self, x):
        if self.m == 0:
            return np.empty((0, self.n))
        x = np.asarray(x, dtype=float)
        ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
        gx = []
        for index in np.flatnonzero(ieq):
            _, g_index = self.cons(x, index, True)
            gx.append(g_index)
        gx = np.array(gx, dtype=float)
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
        aub = np.array(aub, dtype=float)
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
            beq.append(c_index - 0.5 * (self.cl[index] + self.cu[index]))
        aeq = np.array(aeq, dtype=float)
        return np.reshape(aeq, (-1, self.n)), np.array(beq)

    def _project_x0(self):
        if self.m == 0:
            self.x0 = np.minimum(self.xu, np.maximum(self.xl, self.x0))
        elif self.mlub == 0 and self.mleq > 0 and np.all(self.xl == -np.inf) and np.all(self.xu == np.inf):
            self.x0 = lstsq(self.aeq, self.beq - np.dot(self.aeq, self.x0))[0]
        else:
            bounds = Bounds(self.xl, self.xu, True)
            constraints = []
            if self.mlub > 0:
                constraints.append(LinearConstraint(self.aub, -np.inf, self.bub))
            if self.mleq > 0:
                constraints.append(LinearConstraint(self.aeq, self.beq, self.beq))
            if self.mnlub > 0:
                rhs = np.zeros(self.mnlub, dtype=float)
                constraints.append(NonlinearConstraint(self.cub, -np.inf, rhs, self.cub_jac))
            if self.mnleq > 0:
                rhs = np.zeros(self.mnleq, dtype=float)
                constraints.append(NonlinearConstraint(self.ceq, rhs, rhs, self.ceq_jac))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = minimize(self._distance, self.x0, jac=True, bounds=bounds,
                               constraints=constraints)  # noqa
            self.x0 = np.array(res.x, dtype=float)

    def _distance(self, x):
        x = np.asarray(x, dtype=float)
        fx = 0.5 * np.inner(x - self.x0, x - self.x0)
        gx = x - self.x0
        return fx, gx
