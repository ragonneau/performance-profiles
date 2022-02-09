import os
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.backends import backend_pdf
from matplotlib.ticker import MultipleLocator

from optimize import Minimizer
from problem import CUTEstProblems

BASE_DIR = Path(__file__).resolve(strict=True).parent
ARCH_DIR = Path(BASE_DIR, os.environ.get('PYCUTEST_CACHE'))


class Profiles:

    def __init__(self, n_max, n_min=1, feature='plain', constraints='U', callback=None):
        self._n_min = n_min
        self._n_max = n_max
        self._feature = feature
        self._constraints = constraints

        self._perf_dir = Path(ARCH_DIR, 'performance', f'{self._n_min}-{self._n_max}')
        self._data_dir = Path(ARCH_DIR, 'data', f'{self._n_min}-{self._n_max}')
        self._eval_dir = Path(ARCH_DIR, 'storage')

        self._problems = CUTEstProblems(self._n_min, self._n_max, self._constraints, callback)
        print()
        print(f'*** {len(self._problems)} problem(s) loaded ***')
        print()
        if len(self._problems) == 0:
            raise RuntimeError

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    def __call__(self, solvers, options=None, load=True, **kwargs):
        solvers = list(map(str.lower, solvers))
        options = self.set_default_options(options)

        self.create_arborescence()
        self.profiles(solvers, options, load, **kwargs)

    def set_default_options(self, options):
        if options is None:
            options = {}
        options = dict(options)
        options.setdefault('maxfev', 500 * self._n_max)
        return options

    def create_arborescence(self):
        self._perf_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._eval_dir.mkdir(parents=True, exist_ok=True)

    def get_storage_path(self, problem, solver):
        if problem.sifParams is None:
            cache = Path(self._eval_dir, problem.name)
        else:
            sif = '_'.join(f'{k}{v}' for k, v in problem.sifParams.items())
            cache = Path(self._eval_dir, f'{problem.name}_{sif}')
        cache.mkdir(exist_ok=True)
        obj_path = Path(cache, f'obj-hist-{solver.lower()}.npy')
        mcv_path = Path(cache, f'mcv-hist-{solver.lower()}.npy')
        var_path = Path(cache, f'var-hist-{solver.lower()}.npy')
        return obj_path, mcv_path, var_path

    def get_performance_path(self, solvers):
        solvers = '_'.join(sorted(solvers))
        filename = f'perf-{solvers}-{self._constraints}.pdf'
        return Path(self._perf_dir, filename)

    def get_data_path(self, solvers):
        solvers = '_'.join(sorted(map(str.lower, solvers)))
        filename = f'data-{solvers}-{self._constraints}.pdf'
        return Path(self._data_dir, filename)

    def profiles(self, solvers, options=None, load=True, **kwargs):
        options = self.set_default_options(options)
        merits = self.run_solvers(solvers, options, load, **kwargs)

        penalty = 2
        dpi = 200
        maxfev = merits.shape[-1]
        data_y = int(1.2 * maxfev)
        styles = ['-', '--', '-.', ':']

        solutions = np.nanmin(merits, axis=(1, 2))
        f0 = merits[..., 0]
        pdf_perf = backend_pdf.PdfPages(self.get_performance_path(solvers))
        pdf_data = backend_pdf.PdfPages(self.get_data_path(solvers))
        print()
        for prec in range(1, 10):
            print(f'Creating plots with tau = 1e-{prec}.')
            tau = 10 ** (-prec)

            work = np.full_like(f0, np.nan)
            for i, problem in enumerate(self._problems):
                for j, solver in enumerate(solvers):
                    rho = tau * f0[i, j] + (1.0 - tau) * solutions[i]
                    if not np.all(np.isnan(merits[i, j, :])) and np.nanmin(merits[i, j, :]) <= rho:
                        work[i, j] = np.argmax(merits[i, j, :] <= rho) + 1

            perf = np.full_like(f0, np.nan)
            for i, problem in enumerate(self._problems):
                if not np.all(np.isnan(work[i, :])):
                    perf[i, :] = work[i, :] / np.nanmin(work[i, :])
            perf = np.log2(perf)
            ratio_max = np.nanmax(perf, initial=0.0)
            perf[np.isnan(perf)] = penalty * ratio_max

            data = np.full((len(solvers), data_y), np.nan, dtype=float)
            for j in range(len(solvers)):
                for k in range(maxfev):
                    data[j, k] = np.where(work[:, j] <= k)[0].size
                data[j, maxfev:] = data[j, maxfev - 1]
            data /= len(self._problems)

            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111)
            ax.yaxis.tick_left()
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(direction='in', which='both')
            for j, solver in enumerate(solvers):
                perf[:, j] = np.sort(perf[:, j])
                x = np.repeat(perf[:, j], 2)[1:]
                y = np.repeat(np.linspace(1 / len(self._problems), 1, len(self._problems)), 2)[:-1]
                x = np.r_[0, x[0], x, penalty * ratio_max]
                y = np.r_[0, 0, y, y[-1]]
                plt.plot(x, y, styles[j % len(styles)], label=solvers[j], linewidth=1)
            plt.xlim(0, 1.1 * ratio_max)
            plt.ylim(0, 1.1)
            plt.xlabel(r'$\log_2(\mathrm{NF}/\mathrm{NF}_{\min})$')
            plt.ylabel(fr'Performance profile ($\tau=10^{{-{prec}}}$)')
            plt.legend(loc='lower right')
            pdf_perf.savefig(fig, bbox_inches='tight')
            plt.close()

            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111)
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(direction='in', which='both')
            for j, solver in enumerate(solvers):
                x = np.repeat(np.linspace(0, data_y / (self._n_max + 1), data_y), 2)[1:]
                y = np.repeat(data[j, :], 2)[:-1]
                plt.plot(x, y, styles[j % len(styles)], label=solvers[j], linewidth=1)
            plt.xlim(0, 1.1 * ratio_max)
            plt.ylim(0, 1.1)
            plt.xlabel(r'$\mathrm{NF}/(n+1)$')
            plt.ylabel(fr'Data profile ($\tau=10^{{-{prec}}}$)')
            plt.legend(loc='lower right')
            pdf_data.savefig(fig, bbox_inches='tight')
            plt.close()
        pdf_perf.close()
        pdf_data.close()

    def run_solvers(self, solvers, options=None, load=True, **kwargs):
        options = self.set_default_options(options)
        maxfev = options.get('maxfev')
        merits = np.empty((len(self._problems), len(solvers), maxfev), dtype=float)
        print(('+' + '-' * 15) * 5 + '+')
        print(f'|{"PROBLEM":^15}|{"SOLVER":^15}|{"OBJECTIVE":^15}|{"MAXCV":^15}|{"NFEV":^15}|')
        print(('+' + '-' * 15) * 5 + '+')
        histories = Parallel(n_jobs=-1)(
            self._run_problem_solver(problem, solver, options, load, **kwargs)
            for problem, solver in product(self._problems, solvers))
        print(('+' + '-' * 15) * 5 + '+')
        for i in range(len(self._problems)):
            for j in range(len(solvers)):
                history, nfev = histories[i * len(solvers) + j]
                merits[i, j, :nfev] = history[:nfev]
                merits[i, j, nfev:] = merits[i, j, nfev - 1]
        return merits

    @delayed
    def _run_problem_solver(self, problem, solver, options, load, **kwargs):
        obj_path, mcv_path, var_path = self.get_storage_path(problem, solver)
        maxfev = options.get('maxfev')
        huge = np.finfo(float).max
        merits, nfev = None, 0
        if load and obj_path.is_file() and mcv_path.is_file() and var_path.is_file():
            obj_hist = np.load(obj_path)  # noqa
            mcv_hist = np.load(mcv_path)  # noqa
            merits = self._merit(obj_hist, mcv_hist, huge, **kwargs)
            nfev = merits.size
            if maxfev > nfev:
                var = np.load(var_path)  # noqa
                if var[0] == 0:
                    merits = np.r_[merits, np.full(maxfev - nfev, huge, dtype=float)]
                else:
                    merits, nfev = None, 0
            elif maxfev < nfev:
                nfev = maxfev
                merits = merits[:nfev]
        if merits is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                minimizer = Minimizer(problem, solver, options)
                res, obj_hist, mcv_hist = minimizer()
            nfev = min(obj_hist.size, maxfev)
            merits = np.full(maxfev, huge)
            merits[:nfev] = self._merit(obj_hist[:nfev], mcv_hist[:nfev], huge, **kwargs)
            np.save(obj_path, obj_hist[:nfev])  # noqa
            np.save(mcv_path, mcv_hist[:nfev])  # noqa
            np.save(var_path, np.array([res.status]))  # noqa
        i_min = np.argmin(merits[:nfev])
        obj_i_min = np.format_float_scientific(obj_hist[i_min], 4, False, sign=True, exp_digits=3)
        mcv_i_min = np.format_float_scientific(mcv_hist[i_min], 4, False, sign=True, exp_digits=3)
        print(f'| {problem.name:<14}| {solver:<14}|  {obj_i_min} |  {mcv_i_min} |{nfev:>14} |')
        return merits, nfev

    @staticmethod
    def _merit(obj_hist, mcv_hist, huge, **kwargs):
        merits = np.empty_like(obj_hist)
        for i in range(merits.size):
            if mcv_hist[i] <= kwargs.get('eta1', 1e-13):
                merits[i] = obj_hist[i]
            elif mcv_hist[i] >= kwargs.get('eta2', 1e-5):
                merits[i] = huge
            else:
                merits[i] = obj_hist[i] + kwargs.get('eta3', 1e6) * mcv_hist[i]
            merits[i] = np.nan_to_num(merits[i], nan=huge)
        return merits
