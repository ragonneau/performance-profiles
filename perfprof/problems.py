import re
import warnings
from subprocess import DEVNULL, PIPE, Popen

import numpy as np
import pycutest
from joblib import Parallel, delayed
from scipy.linalg import lstsq
from scipy.optimize import Bounds, LinearConstraint, minimize

EXCLUDED = {
    # The compilation of the sources is prohibitively time-consuming.
    'ARGLALE', 'ARGLBLE', 'ARGLCLE', 'BA-L16LS', 'BA-L21', 'BA-L21LS', 'BA-L49',
    'BA-L49LS', 'BA-L52', 'BA-L52LS', 'BA-L73', 'BA-L73LS', 'BDRY2', 'CHANDHEU',
    'CHARDIS0', 'CHARDIS1', 'DANWOODLS', 'DMN15102', 'DMN15102LS', 'DMN15103',
    'DMN15103LS', 'DMN15332', 'DMN15332LS', 'DMN15333', 'DMN15333LS',
    'DMN37142', 'DMN37142LS', 'DMN37143', 'DMN37143LS', 'GAUSS1LS', 'GAUSS2LS',
    'GAUSS3LS', 'GAUSSELM', 'GOFFIN', 'GPP', 'KOEBHELB', 'LEUVEN3', 'LEUVEN4',
    'LEUVEN5', 'LEUVEN6', 'LHAIFAM', 'LINCONT', 'LIPPERT2', 'LOBSTERZ',
    'MGH17LS', 'MISRA1ALS', 'MISRA1CLS', 'MODEL', 'NASH', 'NELSONLS',
    'OSBORNEA', 'PDE1', 'PDE2', 'PENALTY3', 'RAT43LS', 'RDW2D51F', 'RDW2D51U',
    'RDW2D52B', 'RDW2D52F', 'RDW2D52U', 'ROSEPETAL', 'TWOD', 'WALL100',
    'YATP1SQ', 'YATP2SQ',

    # The starting points contain NaN values.
    'LHAIFAM',

    # The problems contain a lot of NaN.
    'HS62', 'HS112', 'LIN',

    # The problems seem not lower-bounded.
    'INDEF',

    # The problems are known infeasible.
    'ARGLALE', 'ARGLBLE', 'ARGLCLE', 'MODEL', 'NASH',

    # The problems seem infeasible.
    'ANTWERP', 'CRESC4', 'CRESC50', 'DEGENLPA', 'DEGENLPB', 'DIXCHLNG',
    'DUALC1', 'DUALC2', 'DUALC5', 'DUALC8', 'ELATTAR', 'GOULDQP1', 'HIMMELBJ',
    'HONG', 'HS8', 'HS13', 'HS19', 'HS55', 'HS63', 'HS64', 'HS72', 'HS73',
    'HS84', 'HS86', 'HS88', 'HS89', 'HS92', 'HS101', 'HS102', 'HS103', 'HS106',
    'HS107', 'HS109', 'HS119', 'LOADBAL', 'LOTSCHD', 'LSNNODOC', 'PORTFL1',
    'PORTFL2', 'PORTFL3', 'PORTFL4', 'PORTFL6', 'SNAKE', 'SUPERSIM', 'TAME',
    'WACHBIEG',

    # The projection of the initial guess fails.
    'LINCONT',

    # Classical UOBYQA and COBYLA suffer from infinite cycling.
    'GAUSS1LS', 'GAUSS2LS', 'GAUSS3LS', 'MGH17LS', 'MISRA1ALS', 'MISRA1CLS',
    'NELSONLS', 'OSBORNEA', 'RAT43LS',

    # Classical COBYLA suffers from infinite cycling.
    'DANWOODLS', 'KOEBHELB',
}


class Problems(list):
    """
    Wrapper for a sequence of CUTEst problems.
    """

    def __init__(self, n_min, n_max, constraints, callback=None):
        """
        Load the sequence of CUTEst problems.

        Parameters
        ----------
        n_min : int
            Lower bound on the dimension of the problems to consider.
        n_max : int
            Upper bound on the dimension of the problems to consider.
        constraints : str
            Classification code of the problem's constraints.
        callback : callable, optional
            Extra tests to perform on the problem.

                ``callback(problem) -> bool``
        """
        super().__init__()
        self._n_min = n_min
        self._n_max = n_max
        pbs = pycutest.find_problems(
            objective='CLQSO',
            constraints=constraints,
            regular=True,
            n=[self._n_min, self._n_max],
            userM=False,
        )
        attempts = Parallel(n_jobs=-1)(
            self._load(sorted(pbs), i) for i in range(len(pbs)))
        for problem in attempts:
            if problem is not None:
                self.append(problem, callback)

    def append(self, problem, callback=None):
        """
        Append a problem to the current structure.

        Parameters
        ----------
        problem : Problem
            Problem to be appended to the current structure.
        callback : callable, optional
            Extra tests to perform on the problem.

                ``callback(problem) -> bool``
        """
        if self._validate(problem, callback):
            super().append(problem)

    @delayed
    def _load(self, names, i):
        """
        Load a given problem.

        Parameters
        ----------
        names : list
            Names of all problems to be loaded.
        i : int
            Index of the problem to be loaded.

        Returns
        -------
        Problem
            Loaded problem. The return is set to None if the loading failed.
        """
        if names[i] not in EXCLUDED:
            try:
                print(f'Loading {names[i]} ({i + 1}/{len(names)}).')
                if pycutest.problem_properties(names[i])['n'] is None:
                    sif = self._sif_decode(names[i])
                    mask = (sif >= self._n_min) & (sif <= self._n_max)
                    if np.any(mask):
                        return Problem(
                            problemName=names[i],
                            sifParams={'N': np.max(sif[mask])},
                        )
                else:
                    return Problem(problemName=names[i])
            except:
                pass

    def _validate(self, problem, callback=None):
        """
        Validate a given problem.

        Parameters
        ----------
        problem : Problem
            Problem to validate.
        callback : callable, optional
            Extra tests to perform on the problem.

                ``callback(problem) -> bool``
        Returns
        -------
        bool
            Whether the problem is valid.
        """
        valid = isinstance(problem, Problem)
        valid = valid and np.all(problem.vartype == 0)
        valid = valid and problem.n >= self._n_min
        valid = valid and problem.n <= self._n_max
        if callback is not None:
            valid = valid and callback(problem)
        return valid

    @staticmethod
    def _sif_decode(name, param='N'):
        """
        List the available SIF parameters of a given problem.

        Parameters
        ----------
        name : str
            Name of the problem to check.
        param : str, optional
            Name of the parameter to check.

        Returns
        -------
        numpy.ndarray, shape (m,)
            Sorted array of the available SIF parameters.
        """
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


class Problem:
    """
    Wrapper for CUTEst problems.
    """

    def __init__(self, *args, **kwargs):
        """
        Import the CUTEst problem given in the parameters.

        The CUTEst problems are loaded using ``pycutest`` and the parameters
        should comply with those of the ``pycutest.import_problem`` function.
        """
        self._p = pycutest.import_problem(*args, **kwargs)
        self._xl = None
        self._xu = None
        self._aub = None
        self._bub = None
        self._aeq = None
        self._beq = None
        self._project_x0()

    def __getattr__(self, item):
        """
        Get undefined attributes from ``pycutest``.

        Parameters
        ----------
        item : str
            Name of the attribute to get.

        Returns
        -------
        object:
            Attribute loaded from ``pycutest``.

        Raises
        ------
        AttributeError
            The attribute does not exist.
        """
        try:
            return getattr(self._p, item)
        except AttributeError as exc:
            raise AttributeError(item) from exc

    def __getstate__(self):
        """
        Get the state of the current instance.

        This is needed by ``pickle`` to serialize the instance when calling
        instances in a parallel environment.

        Returns
        -------
        dict
            State of the current instance.
        """
        return self.__dict__

    def __setstate__(self, state):
        """
        Set the state of the current instance.

        This is needed by ``pickle`` to serialize the instance when calling
        instances in a parallel environment.

        Parameters
        ----------
        state: dict
            New state of the current instance.
        """
        self.__dict__.update(state)

    def __str__(self):
        """
        Get a string representation of the current instance.

        Returns
        -------
            String representation of the current instance.
        """
        return str(self._p)

    @property
    def xl(self):
        """
        Lower-bound constraints on the decision variables.

        Returns
        -------
        numpy.ndarray, shape(n,)
            Lower-bound constraints on the decision variables.
        """
        if self._xl is not None:
            return self._xl
        self._xl = np.array(self.bl, dtype=float)
        self._xl[self._xl <= -1e20] = -np.inf
        return self._xl

    @property
    def xu(self):
        """
        Upper-bound constraints on the decision variables.

        Returns
        -------
        numpy.ndarray, shape(n,)
            Upper-bound constraints on the decision variables.
        """
        if self._xu is not None:
            return self._xu
        self._xu = np.array(self.bu, dtype=float)
        self._xu[self._xu >= 1e20] = np.inf
        return self._xu

    @property
    def aub(self):
        """
        Jacobian matrix of the linear inequality constraints.

        Returns
        -------
        numpy.ndarray, shape(mlub, n)
            Jacobian matrix of the linear inequality constraints.
        """
        if self._aub is not None:
            return self._aub
        self._aub, self._bub = self._linear_ub()
        return self._aub

    @property
    def bub(self):
        """
        Right-hand side vector of the linear inequality constraints.

        Returns
        -------
        numpy.ndarray, shape (mlub,)
            Right-hand side vector of the linear inequality constraints.
        """
        if self._bub is not None:
            return self._bub
        self._aub, self._bub = self._linear_ub()
        return self._bub

    @property
    def aeq(self):
        """
        Jacobian matrix of the linear equality constraints.

        Returns
        -------
        numpy.ndarray, shape(mlub, n)
            Jacobian matrix of the linear equality constraints.
        """
        if self._aeq is not None:
            return self._aeq
        self._aeq, self._beq = self._linear_eq()
        return self._aeq

    @property
    def beq(self):
        """
        Right-hand side vector of the linear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (mlub,)
            Right-hand side vector of the linear equality constraints.
        """
        if self._beq is not None:
            return self._beq
        self._aeq, self._beq = self._linear_eq()
        return self._beq

    @property
    def mlub(self):
        """
        Number of linear inequality constraints.

        Returns
        -------
        int
            Number of linear inequality constraints.
        """
        if self.m == 0:
            return 0
        iub = self.is_linear_cons & np.logical_not(self.is_eq_cons)
        iub_lower = self.cl[iub] > -1e20
        iub_upper = self.cu[iub] < 1e20
        return np.count_nonzero(iub_lower) + np.count_nonzero(iub_upper)

    @property
    def mleq(self):
        """
        Number of linear equality constraints.

        Returns
        -------
        int
            Number of linear equality constraints.
        """
        if self.m == 0:
            return 0
        ieq = self.is_linear_cons & self.is_eq_cons
        return np.count_nonzero(ieq)

    @property
    def mnlub(self):
        """
        Number of nonlinear inequality constraints.

        Returns
        -------
        int
            Number of nonlinear inequality constraints.
        """
        if self.m == 0:
            return 0
        iub = np.logical_not(self.is_linear_cons | self.is_eq_cons)
        iub_lower = self.cl[iub] > -1e20
        iub_upper = self.cu[iub] < 1e20
        return np.count_nonzero(iub_lower) + np.count_nonzero(iub_upper)

    @property
    def mnleq(self):
        """
        Number of nonlinear equality constraints.

        Returns
        -------
        int
            Number of nonlinear equality constraints.
        """
        if self.m == 0:
            return 0
        ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
        return np.count_nonzero(ieq)

    @property
    def type(self):
        """
        Classification code of the problem's constraints.

        For more information, see the CUTEst classification scheme at
        <https://www.cuter.rl.ac.uk/Problems/classification.shtml>.

        Returns
        -------
        str
            Classification code of the problem's constraints.
        """
        properties = pycutest.problem_properties(self.name)
        return properties.get('constraints')

    def fun(self, x, callback=None, *args, **kwargs):
        """
        Evaluate the objective function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the objective function is to be evaluated.
        callback : callable, optional
            Modifier of the objective function values.

                ``callback(x, fx, *args, **kwargs) -> float``

            where ``fx`` is the true value of the objective function, and `args`
            and `kwargs` are extra parameters to specify the callback function.
            This is usually used to include noise into the objective function.

        Returns
        -------
        float
            Objective function value at `x`.
        """
        x = np.asarray(x, dtype=float)
        fx = self.obj(x)
        if callback is not None:
            fx = callback(x, fx, *args, **kwargs)
        return fx

    def hess(self, x):
        """
        Evaluate the Hessian matrix of the objective function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the Hessian matrix is to be evaluated.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            Hessian matrix of the objective function at `x`.
        """
        x = np.asarray(x, dtype=float)
        return self.hess(x)

    def cub(self, x):
        """
        Evaluate the nonlinear inequality constraint function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the constraint function is to be evaluated.

        Returns
        -------
        numpy.ndarray, shape (mnlub,)
            Constraint function value at `x`.
        """
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
        """
        Evaluate the Jacobian matrix of the nonlinear inequality constraint
        function (each row being the gradient of a constraint).

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the Jacobian matrix is to be evaluated.

        Returns
        -------
        numpy.ndarray, shape (mnlub, n)
            Jacobian matrix of the constraint function at `x`.
        """
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
        """
        Evaluate the nonlinear equality constraint function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the constraint function is to be evaluated.

        Returns
        -------
        numpy.ndarray, shape (mnleq,)
            Constraint function value at `x`.
        """
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
        """
        Evaluate the Jacobian matrix of the nonlinear equality constraint
        function (each row being the gradient of a constraint).

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the Jacobian matrix is to be evaluated.

        Returns
        -------
        numpy.ndarray, shape (mnleq, n)
            Jacobian matrix of the constraint function at `x`.
        """
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
        """
        Evaluate the constraint violation of a given point.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the constraint violation is to be evaluated.

        Returns
        -------
        float
            Constraint violation at `x`.
        """
        violmx = np.max(self.xl - x, initial=0.0)
        violmx = np.max(x - self.xu, initial=violmx)
        violmx = np.max(np.dot(self.aub, x) - self.bub, initial=violmx)
        violmx = np.max(np.abs(np.dot(self.aeq, x) - self.beq), initial=violmx)
        violmx = np.max(self.cub(x), initial=violmx)
        violmx = np.max(np.abs(self.ceq(x)), initial=violmx)
        return violmx

    def _linear_ub(self):
        """
        Build the linear inequality constraints.

        Returns
        -------
        numpy.ndarray, shape (mlub, n)
            Jacobian matrix of the linear inequality constraints.
        numpy.ndarray, shape (mlub,)
            Right-hand side vector of the linear inequality constraints.
        """
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
        """
        Build the linear equality constraints.

        Returns
        -------
        numpy.ndarray, shape (mleq, n)
            Jacobian matrix of the linear equality constraints.
        numpy.ndarray, shape (mleq,)
            Right-hand side vector of the linear equality constraints.
        """
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
        """
        Project the initial guess onto the feasible polyhedron.
        """
        if self.m == 0:
            self.x0 = np.minimum(self.xu, np.maximum(self.xl, self.x0))
        elif self.mlub == 0 and self.mleq > 0 and \
                np.all(self.xl == -np.inf) and np.all(self.xu == np.inf):
            self.x0 += lstsq(self.aeq, self.beq - np.dot(self.aeq, self.x0))[0]
        else:
            bnds = Bounds(self.xl, self.xu, True)
            ctrs = []
            if self.mlub > 0:
                ctrs.append(LinearConstraint(self.aub, -np.inf, self.bub))
            if self.mleq > 0:
                ctrs.append(LinearConstraint(self.aeq, self.beq, self.beq))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = minimize(self._distsq, self.x0, jac=True, bounds=bnds,
                               constraints=ctrs)  # noqa
            self.x0 = np.array(res.x, dtype=float)

    def _distsq(self, x):
        """
        Evaluate the objective function of the projection problem.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the distance is to be evaluated.

        Returns
        -------
        float
            Objective function value at `x`.
        numpy.ndarray, shape (n,)
            Gradient of the objective function of the projection problem at `x`.
        """
        x = np.asarray(x, dtype=float)
        fx = 0.5 * np.inner(x - self.x0, x - self.x0)
        gx = x - self.x0
        return fx, gx
