import numpy as np
import pdfo
from cobyqa import minimize as cobyqa
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize


class Minimizer:
    """
    Wrapper for minimizing problems.
    """

    def __init__(self, problem, solver, options, callback, *args, **kwargs):
        """
        Set up the minimization of a problem using a given solver.

        Parameters
        ----------
        problem : Problem
            Problem to be minimized by the solver.
        solver : str
            Solver to employ to minimize the problem.
        options : dict
            Options to forward to the solver.
        callback : callable
            Modifier of the objective function values.

                ``callback(x, fx, *args, **kwargs) -> float``

            where ``x`` is the evaluation point, ``fx`` is the true value of the
            objective function at ``x``, and `args` and `kwargs` are extra
            parameters to specify the callback function. This is usually used to
            include noise into the objective function.

        Raises
        ------
        NotImplementedError
            The arguments are inconsistent.
        """
        self._prb = problem
        self._slv = solver
        self._opts = dict(options)
        self._callback = callback
        self._args = args
        self._kwargs = kwargs
        self._obj_hist = []
        self._mcv_hist = []
        if not self._validate():
            raise NotImplementedError

    def __call__(self):
        """
        Run the actual computations.

        Returns
        -------
        {scipy.optimize.OptimizeResult, pdfo.OptimizeResult}
            Result structure of the optimization method.
        numpy.ndarray, shape (m,)
            Objective function values encountered by the optimization method.
        numpy.ndarray, shape (m,)
            Constraint violations encountered by the optimization method.
        """
        self._obj_hist = []
        self._mcv_hist = []
        if self._slv.lower() in ['cobyqa', 'cobyqa-simple-tcg']:
            kwargs = {'improve_tcg': self._slv.lower() == 'cobyqa'}
            res = cobyqa(self._eval, self._prb.x0, xl=self._prb.xl,
                         xu=self._prb.xu, Aub=self._prb.aub, bub=self._prb.bub,
                         Aeq=self._prb.aeq, beq=self._prb.beq,
                         cub=self._prb.cub, ceq=self._prb.ceq,
                         options=self._opts, **kwargs)
        elif self._slv.lower() == 'cobyqa-relax':
            identity = np.eye(self._prb.n)
            aub = np.vstack((self._prb.aub, -identity, identity))
            bub = np.r_[self._prb.bub, -self._prb.xl, self._prb.xu]
            res = cobyqa(self._eval, self._prb.x0, Aub=aub, bub=bub,
                         Aeq=self._prb.aeq, beq=self._prb.beq,
                         cub=self._prb.cub, ceq=self._prb.ceq,
                         options=self._opts)
        else:
            bnds = Bounds(self._prb.xl, self._prb.xu)
            ctrs = []
            if self._prb.mlub > 0:
                rhs = self._prb.bub
                ctrs.append(LinearConstraint(self._prb.aub, -np.inf, rhs))
            if self._prb.mleq > 0:
                rhs = self._prb.beq
                ctrs.append(LinearConstraint(self._prb.aeq, rhs, rhs))
            if self._prb.mnlub > 0:
                rhs = np.zeros(self._prb.mnlub, dtype=float)
                ctrs.append(NonlinearConstraint(self._prb.cub, -np.inf, rhs))
            if self._prb.mnleq > 0:
                rhs = np.zeros(self._prb.mnleq, dtype=float)
                ctrs.append(NonlinearConstraint(self._prb.ceq, rhs, rhs))
            kwargs = {
                'fun': self._eval,
                'x0': self._prb.x0,
                'bounds': bnds,
                'constraints': ctrs,
                'options': self._opts,
            }
            if self._slv.lower() != 'pdfo':
                kwargs['method'] = self._slv
            if self._slv.lower() in pdfo.__all__:
                self._opts['eliminate_lin_eq'] = False
                res = pdfo.pdfo(**kwargs)
                if self._prb.type in 'XBLQO':
                    res.maxcv = self._prb.maxcv(res.x)
                if hasattr(res, 'constrviolation'):
                    del res.constrviolation
            else:
                res = minimize(**kwargs)
                if self._prb.type in 'XBLQO':
                    res.maxcv = self._prb.maxcv(res.x)
        obj_hist = np.array(self._obj_hist, dtype=float)
        mcv_hist = np.array(self._mcv_hist, dtype=float)
        return res, obj_hist, mcv_hist

    def _validate(self):
        """
        Validate the given solver.

        Returns
        -------
        bool
            Whether the given solver is valid.
        """
        valid_solvers = {'cobyla', 'cobyqa', 'cobyqa-relax', 'pdfo', 'slsqp'}
        if self._prb.type not in 'QO':
            valid_solvers.update({'lincoa'})
        if self._prb.type not in 'NLQO':
            valid_solvers.update({'cobyqa-simple-tcg', 'bobyqa', 'l-bfgs-b', 'nelder-mead', 'tnc'})
        if self._prb.type not in 'XBNLQO':
            valid_solvers.update({'bfgs', 'cg', 'newuoa', 'uobyqa'})
        valid = self._slv.lower() in valid_solvers
        return valid

    def _eval(self, x):
        """
        Evaluate the objective function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the objective function is to be evaluated.

        Returns
        -------
        float
            Objective function value at `x`.
        """
        fx = self._prb.fun(x, self._callback, *self._args, **self._kwargs)
        if hasattr(fx, '__len__') and len(fx) == 2:
            self._obj_hist.append(fx[0])
            fx = fx[1]
        else:
            self._obj_hist.append(fx)
        self._mcv_hist.append(self._prb.maxcv(x))
        return fx
