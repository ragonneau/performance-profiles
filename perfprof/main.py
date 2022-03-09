import csv
import os
import re
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from cycler import cycler
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.backends import backend_pdf
from matplotlib.ticker import MultipleLocator

from .optimize import Minimizer

os.environ.setdefault('PYCUTEST_CACHE', 'archives')
BASE_DIR = Path.cwd()
ARCH_DIR = Path(BASE_DIR, 'archives')
ARCH_DIR.mkdir(exist_ok=True)
from .problems import Problems  # noqa


class Profiles:
    """
    Wrapper for the performance and data profiles.
    """

    def __init__(self, n_max, n_min=1, feature='plain', constraints='U',
                 callback=None, **kwargs):
        """
        Initialize the computations of the performance and data profiles.

        Parameters
        ----------
        n_max : int
            Upper bound on the dimension of the problems to consider.
        n_min : int, optional
            Lower bound on the dimension of the problems to consider.
        feature : str, optional
            Feature used to modify the objective function. Accepted features are

                - 'plain': the objective function is not modified.
                - 'Lq': the objective function is regularized with a 0.25-norm.
                - 'Lh': the objective function is regularized with a 0.5-norm.
                - 'L1': the objective function is regularized with a 1-norm.
                - 'noisy': the objective function perturbed.
                - 'signif[0-9]+' : only some digits of the objective function
                  are significant (the one given in the feature). The remaining
                  digits are perturbed.
            {'plain', 'lq', 'lh', 'l1', 'noisy', signif???}
        constraints : str, optional
            Classification code of the problem's constraints.
        callback : callable, optional
            Extra tests to perform on the problem.

                ``callback(problem) -> bool``

        Other Parameters
        ----------------
        regularization : float, optional
            Regularization parameter used for regularized features.
        noise_type : {'relative', 'absolute'}, optional
            Noise type used by the noisy feature.
        noise_level : float, optional
            Noise level used by the noisy feature.
        rerun : int, optional
            Number of run performed when using the noisy feature.
        """
        self._n_min = n_min
        self._n_max = n_max
        self._feat = feature.lower()
        self._ctrs = constraints.upper()

        n_string = f'{self._n_min}-{self._n_max}'
        self._perf_dir = Path(ARCH_DIR, 'performance', self._feat, n_string)
        self._data_dir = Path(ARCH_DIR, 'data', self._feat, n_string)
        self._eval_dir = Path(ARCH_DIR, 'storage', self._feat)

        self._feat_opts = self.get_feature_options(**kwargs)
        self._prbs = Problems(self._n_min, self._n_max, self._ctrs, callback)
        print()
        print(f'*** {len(self._prbs)} problem(s) loaded ***')
        print()
        if len(self._prbs) == 0:
            raise RuntimeError

        std_cycle = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                  '#bcbd22', '#17becf'])
        std_cycle += cycler(linestyle=['-', '--', ':', '-.'])
        plt.rc('axes', prop_cycle=std_cycle)
        plt.rc('figure', dpi=200)
        plt.rc('lines', linewidth=1)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    def __call__(self, solvers, options=None, load=True, **kwargs):
        """
        Run the main computations for the performance and data profiles.

        Parameters
        ----------
        solvers : list
            Solvers to use to solve each CUTEst problem.
        options : dict, optional
            Options to forward to each solver.
        load : bool, optional
            Whether to attempt to load the histories.

        Other Parameters
        ----------------
        low_cv : float, optional
            Value of the constraint violation considered to be low.
        high_cv : float, optional
            Value of the constraint violation considered to be high.
        penalty : float, optional
            Penalty coefficient of the merit function
        """
        slvs = list(map(str.lower, solvers))
        opts = self.set_default_options(options)

        self.create_arborescence()
        self.profiles(slvs, opts, load, **kwargs)

    def set_default_options(self, options):
        """
        Set the default options for each solver.

        Parameters
        ----------
        options : dict
            Options provided by the user.

        Returns
        -------
        dict
            Options provided by the user with the default options.
        """
        if options is None:
            options = {}
        opts = dict(options)
        opts.setdefault('maxfev', 500 * self._n_max)
        return opts

    def get_feature_options(self, **kwargs):
        """
        Get the feature options provided by the user and set the default ones.

        Returns
        -------
        dict
            Options for the given feature.

        Other Parameters
        ----------------
        regularization : float, optional
            Regularization parameter used for regularized features.
        noise_type : {'relative', 'absolute'}, optional
            Noise type used by the noisy feature.
        noise_level : float, optional
            Noise level used by the noisy feature.
        rerun : int, optional
            Number of run performed when using the noisy feature.

        Raises
        ------
        NotImplementedError
            The provided features and options are unknown.
        """
        signif = re.match('signif(\d+)', self._feat)
        opts = {'rerun': 1}
        if self._feat in ['lq', 'lh', 'l1']:
            opts['p'] = {
                'lq': 0.25,
                'lh': 0.5,
                'l1': 1.0,
            }.get(self._feat)
            opts['level'] = kwargs.get('regularization', 1.0)
        elif self._feat == 'noisy':
            opts['type'] = str(kwargs.get('noise_type', 'relative'))
            if opts['type'] not in ['relative', 'absolute']:
                raise NotImplementedError
            opts['level'] = float(kwargs.get('noise_level', 1e-3))
            if opts['level'] <= 0.0:
                raise NotImplementedError
            opts['rerun'] = int(kwargs.get('rerun', 10))
            if opts['rerun'] < 0:
                raise NotImplementedError
        elif signif:
            self._feat = 'signif'
            opts['signif'] = int(signif.group(1))
            if opts['signif'] <= 0:
                raise NotImplementedError
        elif self._feat != 'plain':
            raise NotImplementedError
        return opts

    def create_arborescence(self):
        """
        Create the storage arborescence.
        """
        self._perf_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._eval_dir.mkdir(parents=True, exist_ok=True)

    def get_storage_path(self, problem, solver, k):
        """
        Get the storage path for the computations.

        Parameters
        ----------
        problem : perfprof.Problem
            Problem minimized during the computations.
        solver : str
            Solver used to minimize the problem.
        k : int
            Index of the run.

        Returns
        -------
        pathlib.Path
            Path to the objective function value history.
        pathlib.Path
            Path to the constraint violation history.
        pathlib.Path
            Path to the extra variables history.
        """
        if problem.sifParams is None:
            cache = Path(self._eval_dir, problem.name)
        else:
            sif = '_'.join(f'{k}{v}' for k, v in problem.sifParams.items())
            cache = Path(self._eval_dir, f'{problem.name}_{sif}')
        cache.mkdir(exist_ok=True)
        if self._feat_opts['rerun'] == 1:
            obj_path = Path(cache, f'obj-hist-{solver.lower()}.npy')
            mcv_path = Path(cache, f'mcv-hist-{solver.lower()}.npy')
            var_path = Path(cache, f'var-hist-{solver.lower()}.npy')
        else:
            obj_path = Path(cache, f'obj-hist-{solver.lower()}-{k}.npy')
            mcv_path = Path(cache, f'mcv-hist-{solver.lower()}-{k}.npy')
            var_path = Path(cache, f'var-hist-{solver.lower()}-{k}.npy')
        return obj_path, mcv_path, var_path

    def get_perf_path(self, slvs):
        """
        Get the path to the performance profile archive.

        Parameters
        ----------
        slvs : list
            Solvers to use to solve each CUTEst problem.

        Returns
        -------
        pathlib.Path
            Path to the PDF format of the performance profiles.
        pathlib.Path
            Path to the CSV format of the performance profiles.
        pathlib.Path
            Path to the TXT format of the performance profiles.
        """
        slvs = '_'.join(sorted(slvs))
        pdf_path = Path(self._perf_dir, f'perf-{slvs}-{self._ctrs}.pdf')
        csv_path = Path(self._perf_dir, f'perf-{slvs}-{self._ctrs}.csv')
        txt_path = Path(self._perf_dir, f'perf-{slvs}-{self._ctrs}.txt')
        return pdf_path, csv_path, txt_path

    def get_data_path(self, slvs):
        """
        Get the path to the data profile archive.

        Parameters
        ----------
        slvs : list
            Solvers to use to solve each CUTEst problem.

        Returns
        -------
        pathlib.Path
            Path to the PDF format of the data profiles.
        pathlib.Path
            Path to the CSV format of the data profiles.
        pathlib.Path
            Path to the TXT format of the data profiles.
        """
        slvs = '_'.join(sorted(map(str.lower, slvs)))
        pdf_path = Path(self._data_dir, f'data-{slvs}-{self._ctrs}.pdf')
        csv_path = Path(self._data_dir, f'data-{slvs}-{self._ctrs}.csv')
        txt_path = Path(self._data_dir, f'data-{slvs}-{self._ctrs}.txt')
        return pdf_path, csv_path, txt_path

    def profiles(self, solvers, options=None, load=True, **kwargs):
        """
        Evaluate and save the performance and the data profiles using the given
        solvers on each CUTEst problems.

        Parameters
        ----------
        solvers : list
            Solvers to use to solve each CUTEst problem.
        options : dict, optional
            Options to forward to each solver.
        load : bool, optional
            Whether to attempt to load the histories.

        Other Parameters
        ----------------
        low_cv : float, optional
            Value of the constraint violation considered to be low.
        high_cv : float, optional
            Value of the constraint violation considered to be high.
        penalty : float, optional
            Penalty coefficient of the merit function
        """
        opts = self.set_default_options(options)
        merits = self.run_solvers(solvers, opts, load, **kwargs)

        prec_min = 1
        prec_max = 9
        penalty = 2
        ratio_max = 1e-6
        rerun = merits.shape[-2]
        maxfev = merits.shape[-1]

        f0 = np.nanmin(merits[..., 0], 1, initial=np.inf)
        fmin = np.nanmin(merits, (1, 2, 3), initial=np.inf)
        if self._feat in ['signif', 'noisy']:
            rerun_sav = self._feat_opts['rerun']
            feat_sav = self._feat
            self._feat_opts['rerun'] = 1
            self._feat = 'plain'
            merits_plain = self.run_solvers(solvers, opts, load, **kwargs)
            fmin_plain = np.nanmin(merits_plain, (1, 2, 3))
            fmin = np.minimum(fmin, fmin_plain)
            self._feat_opts['rerun'] = rerun_sav
            self._feat = feat_sav

        pdf_perf_path, csv_perf_path, txt_perf_path = self.get_perf_path(solvers)
        pdf_data_path, csv_data_path, txt_data_path = self.get_data_path(solvers)
        raw_col = (prec_max - prec_min + 1) * (len(solvers) + 1)
        raw_perf = np.empty((2 * len(self._prbs) + 2, raw_col), dtype=float)
        raw_data = np.empty((2 * maxfev - 1, raw_col), dtype=float)
        pdf_perf = backend_pdf.PdfPages(pdf_perf_path)
        pdf_data = backend_pdf.PdfPages(pdf_data_path)
        print()
        for prec in range(prec_min, prec_max + 1):
            print(f'Creating plots with tau = 1e-{prec}.')
            tau = 10 ** (-prec)

            work = np.full((len(self._prbs), len(solvers), rerun), np.nan)
            for i in range(len(self._prbs)):
                for j in range(len(solvers)):
                    for k in range(rerun):
                        if np.isfinite(fmin[i]):
                            rho = tau * f0[i, k] + (1.0 - tau) * fmin[i]
                            rho = max(rho, fmin[i])
                        else:
                            rho = -np.inf
                        if np.nanmin(merits[i, j, k, :], initial=np.inf) <= rho:
                            index = np.argmax(merits[i, j, k, :] <= rho)
                            work[i, j, k] = index + 1
            work = np.mean(work, -1)

            perf = np.full((len(self._prbs), len(solvers)), np.nan, dtype=float)
            for i in range(len(self._prbs)):
                if not np.all(np.isnan(work[i, :])):
                    perf[i, :] = work[i, :] / np.nanmin(work[i, :])
            perf = np.log2(perf)
            ratio_max = np.nanmax(perf, initial=ratio_max)
            perf[np.isnan(perf)] = penalty * ratio_max
            perf = np.sort(perf, 0)

            data = np.full((len(solvers), maxfev), np.nan, dtype=float)
            for j in range(len(solvers)):
                for k in range(maxfev):
                    data[j, k] = np.count_nonzero(work[:, j] <= k + 1)
            data /= len(self._prbs)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.yaxis.tick_left()
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(direction='in', which='both')
            y = np.linspace(1 / len(self._prbs), 1, len(self._prbs))
            y = np.repeat(y, 2)[:-1]
            y = np.r_[0, 0, y, y[-1]]
            i_col = (prec - 1) * (len(solvers) + 1)
            raw_perf[:, i_col] = y
            for j, solver in enumerate(solvers):
                x = np.repeat(perf[:, j], 2)[1:]
                x = np.r_[0, x[0], x, penalty * ratio_max]
                raw_perf[:, i_col + j + 1] = x
                plt.plot(x, y, label=solvers[j])
            plt.xlim(0, 1.1 * ratio_max)
            plt.ylim(0, 1)
            plt.xlabel(r'$\log_2(\mathrm{NF}/\mathrm{NF}_{\min})$')
            plt.ylabel(fr'Performance profile ($\tau=10^{{-{prec}}}$)')
            plt.legend(loc='lower right')
            pdf_perf.savefig(fig, bbox_inches='tight')
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(direction='in', which='both')
            x = np.linspace(0, maxfev / (self._n_max + 1), maxfev)
            x = np.repeat(x, 2)[1:]
            raw_data[:, i_col] = x
            for j, solver in enumerate(solvers):
                y = np.repeat(data[j, :], 2)[:-1]
                raw_data[:, i_col + j + 1] = y
                plt.plot(x, y, label=solvers[j])
            plt.xlim(0, 1.1 * ratio_max)
            plt.ylim(0, 1)
            plt.xlabel(r'$\mathrm{NF}/(n+1)$')
            plt.ylabel(fr'Data profile ($\tau=10^{{-{prec}}}$)')
            plt.legend(loc='lower right')
            pdf_data.savefig(fig, bbox_inches='tight')
            plt.close()

        print('Saving performance profiles.')
        pdf_perf.close()
        with open(txt_perf_path, 'w') as fd:
            fd.write('\n'.join(p.name for p in self._prbs))

        print('Saving data profiles.')
        pdf_data.close()
        with open(txt_data_path, 'w') as fd:
            fd.write('\n'.join(p.name for p in self._prbs))

        print('Saving raw performance profiles.')
        with open(csv_perf_path, 'w') as fd:
            csv_perf = csv.writer(fd)
            header_perf = np.array([[f'y{i}', *[f'x{i}_{s}' for s in solvers]]
                                    for i in range(prec_min, prec_max + 1)])
            csv_perf.writerow(header_perf.flatten())
            csv_perf.writerows(raw_perf)

        print('Saving raw data profiles.')
        with open(csv_data_path, 'w') as fd:
            csv_data = csv.writer(fd)
            header_data = np.array([[f'x{i}', *[f'y{i}_{s}' for s in solvers]]
                                    for i in range(prec_min, prec_max + 1)])
            csv_data.writerow(header_data.flatten())
            csv_data.writerows(raw_data)

    def run_solvers(self, solvers, options=None, load=True, **kwargs):
        """
        Solve each CUTEst problem using each solver in the parameters.

        Parameters
        ----------
        solvers : list
            Solvers to use to solve each CUTEst problem.
        options : dict, optional
            Options to forward to each solver.
        load : bool, optional
            Whether to attempt to load the histories.

        Returns
        -------
        numpy.ndarray, shape (n_prbs, n_slvs, rerun, maxfev)
            Merit values returned by the solvers.

        Other Parameters
        ----------------
        low_cv : float, optional
            Value of the constraint violation considered to be low.
        high_cv : float, optional
            Value of the constraint violation considered to be high.
        penalty : float, optional
            Penalty coefficient of the merit function
        """
        opts = self.set_default_options(options)
        maxfev = opts.get('maxfev')
        rerun = self._feat_opts['rerun']

        merits = np.empty((len(self._prbs), len(solvers), rerun, maxfev),
                          dtype=float)
        self._print_header('PROBLEM', 'SOLVER', 'OBJECTIVE', 'MAXCV', 'NFEV')
        histories = Parallel(n_jobs=-1)(
            self._run_problem_solver(prb, slv, k, opts, load, **kwargs)
            for prb, slv, k in product(self._prbs, solvers, range(rerun)))
        self._print_footer(5)
        for i in range(len(self._prbs)):
            for j in range(len(solvers)):
                for k in range(rerun):
                    index = (i * len(solvers) + j) * rerun + k
                    history, nfev = histories[index]
                    merits[i, j, k, :nfev] = history[:nfev]
                    merits[i, j, k, nfev:] = merits[i, j, k, nfev - 1]
        return merits

    @delayed
    def _run_problem_solver(self, problem, solver, k, options, load, **kwargs):
        """
        Solve a given problem with a given solver.

        Parameters
        ----------
        problem : perfprof.Problem
            Problem to be solved.
        solver : str
            Solver to employ to solve the problem.
        k : int
            Index of the run.
        options : dict
            Options to forward to the solver.
        load : bool
            Whether to attempt to load the histories.

        Returns
        -------
        numpy.ndarray, shape (maxfev,)
            Merit function values provided by the optimization method.
        int
            Number of function evaluations

        Other Parameters
        ----------------
        low_cv : float, optional
            Value of the constraint violation considered to be low.
        high_cv : float, optional
            Value of the constraint violation considered to be high.
        penalty : float, optional
            Penalty coefficient of the merit function
        """
        obj_path, mcv_path, var_path = self.get_storage_path(problem, solver, k)
        maxfev = options.get('maxfev')
        merits, nfev = None, 0
        if load and obj_path.is_file() and mcv_path.is_file() and \
                var_path.is_file():
            obj_hist = np.load(obj_path)  # noqa
            mcv_hist = np.load(mcv_path)  # noqa
            merits = self._merit(obj_hist, mcv_hist, **kwargs)
            nfev = merits.size
            if maxfev > nfev:
                var = np.load(var_path)  # noqa
                if var[0]:
                    remain = np.full(maxfev - nfev, np.nan, dtype=float)
                    merits = np.r_[merits, remain]
                else:
                    merits, nfev = None, 0
            elif maxfev < nfev:
                nfev = maxfev
                merits = merits[:nfev]
        if merits is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                opt = Minimizer(problem, solver, options, self._noise, k=k)
                res, obj_hist, mcv_hist = opt()
            nfev = min(obj_hist.size, maxfev)
            merits = np.full(maxfev, np.nan)
            merits[:nfev] = self._merit(obj_hist[:nfev], mcv_hist[:nfev],
                                        **kwargs)
            np.save(obj_path, obj_hist[:nfev])  # noqa
            np.save(mcv_path, mcv_hist[:nfev])  # noqa
            np.save(var_path, np.array([res.success]))  # noqa
        if not np.all(np.isnan(merits[:nfev])):
            i = np.nanargmin(merits[:nfev])
        else:
            i = np.nanargmin(mcv_hist[:nfev])
        self._print(problem.name, solver, obj_hist[i], mcv_hist[i], nfev)
        return merits, nfev

    @staticmethod
    def _merit(obj_hist, mcv_hist, **kwargs):
        """
        Evaluate the merit function for comparing the solvers.

        Parameters
        ----------
        obj_hist : numpy.ndarray, shape (m,)
            Objective function values encountered by the optimization method.
        mcv_hist : numpy.ndarray, shape (m,)
            Constraint violations encountered by the optimization method.

        Returns
        -------
        numpy.ndarray, shape (m,)
            Merit function values encountered by the optimization method.

        Other Parameters
        ----------------
        low_cv : float, optional
            Value of the constraint violation considered to be low.
        high_cv : float, optional
            Value of the constraint violation considered to be high.
        penalty : float, optional
            Penalty coefficient of the merit function
        """
        merits = np.empty_like(obj_hist)
        for i in range(merits.size):
            if mcv_hist[i] <= kwargs.get('low_cv', 1e-12):
                merits[i] = obj_hist[i]
            elif mcv_hist[i] >= kwargs.get('high_cv', 1e-2):
                merits[i] = np.nan
            else:
                penalty = kwargs.get('penalty', 1e3)
                merits[i] = obj_hist[i] + penalty * mcv_hist[i]
        return merits

    def _noise(self, x, fx, k=0):
        """
        Modify objective function evaluations according to the feature.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the objective function has been evaluated.
        fx : float
            Objective function value at `x`.
        k : int, optional
            Index of the run.

        Returns
        -------
        float
            Perturbed objective function value.
        """
        if self._feat in ['lq', 'lh', 'l1']:
            p = self._feat_opts['p']
            penalty = self._feat_opts['level']
            fx += penalty * np.linalg.norm(x, p)
        elif self._feat == 'noisy':
            rng = np.random.default_rng(int(1e8 * abs(
                np.sin(k) + np.sin(self._feat_opts['level']) + np.sum(
                    np.sin(np.abs(np.sin(1e8 * x)))))))
            noise = self._feat_opts['level'] * rng.standard_normal()
            if self._feat_opts['type'] == 'absolute':
                fx += noise
            else:
                fx *= 1.0 + noise
        elif self._feat == 'signif' and np.isfinite(fx):
            digits = self._feat_opts['signif']
            if fx == 0.0:
                fx_rounded = 0.0
            else:
                fx_rounded = round(fx, digits - int(
                    np.floor(np.log10(np.abs(fx)))) - 1)
            fx = fx_rounded + (fx - fx_rounded) * np.abs(np.sin(
                np.sin(np.sin(digits)) + np.sin(1e8 * fx) + np.sum(
                    np.sin(np.abs(1e8 * x))) + np.sin(x.size)))
        elif self._feat != 'plain':
            raise NotImplementedError
        return fx

    def _print_header(self, *args):
        """
        Print the header of the computations, given in the arguments.
        """
        self._print_footer(len(args))
        print('|' + '|'.join(f'{arg:^15}' for arg in args) + '|')
        self._print_footer(len(args))

    @staticmethod
    def _print_footer(k):
        """
        Print the footer of the computations.

        Parameters
        ----------
        k : int
            Number of columns.
        """
        print(('+' + '-' * 15) * k + '+')

    @staticmethod
    def _print(*args):
        """
        Print a regular line of the computations, given in the arguments.
        """
        line = '|'
        for arg in args:
            if isinstance(arg, int):
                line += f'{arg:>14} |'
            elif isinstance(arg, (float, np.generic)):
                temp = np.format_float_scientific(
                    arg, 4, False, sign=True, exp_digits=3)
                line += f'  {temp} |'
            else:
                line += f' {arg:<14}|'
        print(line)
