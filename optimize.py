import numpy as np
import pdfo
from cobyqa import minimize
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize as scipy_minimize


class Minimizer:

    def __init__(self, problem, solver, options):
        self._problem = problem
        self._solver = solver
        self._options = dict(options)
        self._obj_hist = []
        self._mcv_hist = []
        if not self._validate():
            raise NotImplementedError

    def __call__(self):
        self._obj_hist = []
        self._mcv_hist = []
        if self._solver.lower() == 'cobyqa':
            res = minimize(self._eval, self._problem.x0, xl=self._problem.xl,
                           xu=self._problem.xu, Aub=self._problem.aub,
                           bub=self._problem.bub, Aeq=self._problem.aeq,
                           beq=self._problem.beq, cub=self._problem.cub,
                           ceq=self._problem.ceq, options=self._options)
        else:
            bounds = Bounds(self._problem.xl, self._problem.xu)
            constraints = []
            if self._problem.mlub > 0:
                constraints.append(LinearConstraint(self._problem.aub, -np.inf, self._problem.bub))
            if self._problem.mleq > 0:
                constraints.append(LinearConstraint(self._problem.aeq, self._problem.beq, self._problem.beq))
            if self._problem.mnlub > 0:
                rhs = np.zeros(self._problem.mnlub, dtype=float)
                constraints.append(NonlinearConstraint(self._problem.cub, -np.inf, rhs))
            if self._problem.mnleq > 0:
                rhs = np.zeros(self._problem.mnleq, dtype=float)
                constraints.append(NonlinearConstraint(self._problem.ceq, rhs, rhs))
            if self._solver.lower() in pdfo.__all__:
                self._options['eliminate_lin_eq'] = False
                kwargs = {
                    'fun': self._eval,
                    'x0': self._problem.x0,
                    'bounds': bounds,
                    'constraints': constraints,
                    'options': self._options,
                }
                if self._solver.lower() != 'pdfo':
                    kwargs['method'] = self._solver.lower()
                res = pdfo.pdfo(**kwargs)
                if self._problem.type in 'XBLQO':
                    res.maxcv = self._problem.maxcv(res.x)
                if hasattr(res, 'constrviolation'):
                    del res.constrviolation
            else:
                res = scipy_minimize(self._eval, self._problem.x0,
                                     method=self._solver, bounds=bounds,
                                     constraints=constraints,  # noqa
                                     options=self._options)
                if self._problem.type in 'XBLQO':
                    res.maxcv = self._problem.maxcv(res.x)
        obj_hist = np.array(self._obj_hist, dtype=float)
        mcv_hist = np.array(self._mcv_hist, dtype=float)
        return res, obj_hist, mcv_hist

    def _validate(self):
        valid_solvers = {'cobyla', 'cobyqa', 'pdfo', 'slsqp'}
        if self._problem.type not in 'QO':
            valid_solvers.update({'lincoa'})
        if self._problem.type not in 'NLQO':
            valid_solvers.update({'bobyqa', 'l-bfgs-b', 'nelder-mead', 'tnc'})
        if self._problem.type not in 'XBNLQO':
            valid_solvers.update({'bfgs', 'cg', 'newuoa', 'uobyqa'})
        valid = self._solver.lower() in valid_solvers
        # TODO: Validate options according to corresponding solver.
        return valid

    def _eval(self, x):
        fx = self._problem.fun(x)
        self._obj_hist.append(fx)
        maxcv = self._problem.maxcv(x)
        self._mcv_hist.append(maxcv)
        return fx
