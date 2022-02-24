import csv
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
from problem import Problems

BASE_DIR = Path(__file__).resolve(strict=True).parent
ARCH_DIR = Path(BASE_DIR, os.environ.get('PYCUTEST_CACHE'))


class Profiles:

    def __init__(self, n_max, n_min=1, feature='plain', constraints='U', callback=None):
        self._n_min = n_min
        self._n_max = n_max
        self._feature = feature
        self._constraints = constraints

        n_string = f'{self._n_min}-{self._n_max}'
        self._perf_dir = Path(ARCH_DIR, 'performance', self._feature, n_string)
        self._data_dir = Path(ARCH_DIR, 'data', self._feature, n_string)
        self._eval_dir = Path(ARCH_DIR, 'storage', self._feature)

        self._problems = Problems(self._n_min, self._n_max, self._constraints, callback)
        print()
        print(f'*** {len(self._problems)} problem(s) loaded ***')
        print()
        if len(self._problems) == 0:
            raise RuntimeError('No problem loaded')

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
        pdf = Path(self._perf_dir, f'perf-{solvers}-{self._constraints}.pdf')
        csv = Path(self._perf_dir, f'perf-{solvers}-{self._constraints}.csv')
        return pdf, csv

    def get_data_path(self, solvers):
        solvers = '_'.join(sorted(map(str.lower, solvers)))
        pdf = Path(self._data_dir, f'data-{solvers}-{self._constraints}.pdf')
        csv = Path(self._data_dir, f'data-{solvers}-{self._constraints}.csv')
        return pdf, csv

    def profiles(self, solvers, options=None, load=True, **kwargs):
        options = self.set_default_options(options)
        merits = self.run_solvers(solvers, options, load, **kwargs)

        prec_min = 1
        prec_max = 9
        penalty = 2
        ratio_max = 1e-6
        dpi = 200
        maxfev = merits.shape[-1]
        styles = ['-', '--', '-.', ':']

        f0 = merits[..., 0]
        sols = np.nanmin(merits, axis=(1, 2))

        pdf_perf_path, csv_perf_path = self.get_performance_path(solvers)
        pdf_data_path, csv_data_path = self.get_data_path(solvers)
        n_col = (prec_max - prec_min + 1) * (len(solvers) + 1)
        raw_perf = np.empty((2 * len(self._problems) + 2, n_col), dtype=float)
        raw_data = np.empty((2 * maxfev - 1, n_col), dtype=float)
        pdf_perf = backend_pdf.PdfPages(pdf_perf_path)
        pdf_data = backend_pdf.PdfPages(pdf_data_path)
        print()
        for prec in range(prec_min, prec_max + 1):
            print(f'Creating plots with tau = 1e-{prec}.')
            tau = 10 ** (-prec)

            work = np.full_like(f0, np.nan)
            for i, problem in enumerate(self._problems):
                for j, solver in enumerate(solvers):
                    rho = tau * f0[i, j] + (1.0 - tau) * sols[i]
                    if np.nanmin(merits[i, j, :], initial=np.inf) <= rho:
                        work[i, j] = np.argmax(merits[i, j, :] <= rho) + 1

            perf = np.full_like(f0, np.nan)
            for i, problem in enumerate(self._problems):
                if not np.all(np.isnan(work[i, :])):
                    perf[i, :] = work[i, :] / np.nanmin(work[i, :])
            perf = np.log2(perf)
            ratio_max = np.nanmax(perf, initial=ratio_max)
            perf[np.isnan(perf)] = penalty * ratio_max

            data = np.full((len(solvers), maxfev), np.nan, dtype=float)
            for j in range(len(solvers)):
                for k in range(maxfev):
                    data[j, k] = np.count_nonzero(work[:, j] <= k)
            data /= len(self._problems)

            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111)
            ax.yaxis.tick_left()
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(direction='in', which='both')
            y = np.linspace(1 / len(self._problems), 1, len(self._problems))
            y = np.repeat(y, 2)[:-1]
            y = np.r_[0, 0, y, y[-1]]
            i_col = (prec - 1) * (len(solvers) + 1)
            raw_perf[:, i_col] = y
            for j, solver in enumerate(solvers):
                perf[:, j] = np.sort(perf[:, j])
                x = np.repeat(perf[:, j], 2)[1:]
                x = np.r_[0, x[0], x, penalty * ratio_max]
                raw_perf[:, i_col + j + 1] = x
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
            x = np.linspace(0, maxfev / (self._n_max + 1), maxfev)
            x = np.repeat(x, 2)[1:]
            raw_data[:, i_col] = x
            for j, solver in enumerate(solvers):
                y = np.repeat(data[j, :], 2)[:-1]
                raw_data[:, i_col + j + 1] = y
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

        print(f'Saving raw performance profiles.')
        with open(csv_perf_path, 'w') as fd:
            csv_perf = csv.writer(fd)
            header_perf = np.array([[f'y{i}', *[f'x{i}_{s}' for s in solvers]] for i in range(1, 10)])
            csv_perf.writerow(header_perf.flatten())
            csv_perf.writerows(raw_perf)

        print(f'Saving raw data profiles.')
        with open(csv_data_path, 'w') as fd:
            csv_data = csv.writer(fd)
            header_data = np.array([[f'x{i}', *[f'y{i}_{s}' for s in solvers]] for i in range(1, 10)])
            csv_data.writerow(header_data.flatten())
            csv_data.writerows(raw_data)

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
            if mcv_hist[i] <= kwargs.get('eta1', 1e-12):
                merits[i] = obj_hist[i]
            elif mcv_hist[i] >= kwargs.get('eta2', 1e-2):
                merits[i] = huge
            else:
                merits[i] = obj_hist[i] + kwargs.get('eta3', 1e3) * mcv_hist[i]
            merits[i] = np.nan_to_num(merits[i], nan=huge)
        return merits
