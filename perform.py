import os
import warnings
from pathlib import Path

import cobyqa
import numpy as np
import pdfo
from matplotlib import pyplot as plt
from matplotlib.backends import backend_pdf
from matplotlib.ticker import MultipleLocator

from problem import CUTEstProblems

BASE_DIR = Path(__file__).resolve(strict=True).parent
ARCH_DIR = Path(BASE_DIR, os.environ.get('PYCUTEST_CACHE'))


class Profiles:

    def __init__(self, n, feature='plain', constraints='U', callback=None):
        self.n = n
        self.feature = feature
        self.constraints = constraints

        self.perf_dir = Path(ARCH_DIR, 'performance', str(self.n))
        self.data_dir = Path(ARCH_DIR, 'data', str(self.n))
        self.eval_dir = Path(ARCH_DIR, 'tables')

        self.problems = CUTEstProblems(self.n, self.constraints, callback)
        print()
        print(f'*** {len(self.problems)} problem(s) loaded ***')
        print()

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    def __call__(self, solvers, options=None, load=True):
        solvers = list(map(str.lower, solvers))
        options = self.set_default_options(options)

        self.create_arborescence()
        self.profiles(solvers, options, load)

    def set_default_options(self, options):
        if options is None:
            options = {}
        options = dict(options)
        options.setdefault('maxfev', 500 * self.n)
        return options

    def create_arborescence(self):
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def get_rec_path(self, problem, solver):
        if problem.sifParams is None:
            cache = Path(self.eval_dir, problem.name)
        else:
            sif = '_'.join(f'{k}{v}' for k, v in problem.sifParams.items())
            cache = Path(self.eval_dir, f'{problem.name}_{sif}')
        cache.mkdir(exist_ok=True)
        return Path(cache, f'hist-{solver.lower()}.npy')

    def get_perf_path(self, solvers):
        solvers = '_'.join(sorted(map(str.lower, solvers)))
        filename = f'perf-{solvers}-{self.constraints}.pdf'
        return Path(self.perf_dir, filename)

    def get_data_path(self, solvers):
        solvers = '_'.join(sorted(map(str.lower, solvers)))
        filename = f'data-{solvers}-{self.constraints}.pdf'
        return Path(self.data_dir, filename)

    def profiles(self, solvers, options=None, load=True):
        options = self.set_default_options(options)
        merits = self.run_solvers(solvers, options, load)

        ratio_max = 1e-6
        penalty = 2
        dpi = 200
        maxfev = merits.shape[-1]
        data_y = int(1.2 * maxfev)
        styles = ['-', '--', '-.', ':']

        merits_min = np.min(merits, axis=(1, 2))
        f0 = merits[..., 0]
        pdf_perf = backend_pdf.PdfPages(self.get_perf_path(solvers))
        pdf_data = backend_pdf.PdfPages(self.get_data_path(solvers))
        print()
        for prec in range(1, 10):
            print(f'Creating plots with tau = 1e-{prec}.')
            tau = 10 ** (-prec)

            conv = np.full_like(f0, np.nan)
            for i, problem in enumerate(self.problems):
                for j, solver in enumerate(solvers):
                    rho = tau * f0[i, j] + (1 - tau) * merits_min[i]
                    if np.min(merits[i, j, :]) <= rho:
                        conv[i, j] = np.argmax(merits[i, j, :] <= rho) + 1

            perf = np.full_like(f0, np.nan)
            for i, problem in enumerate(self.problems):
                if not np.all(np.isnan(conv[i, :])):
                    perf[i, :] = conv[i, :] / np.nanmin(conv[i, :])
            perf = np.log2(perf)
            ratio_max = max(ratio_max, np.nanmax(perf))
            perf[np.isnan(perf)] = penalty * ratio_max

            data = np.full((len(solvers), data_y), np.nan)
            for j in range(len(solvers)):
                for k in range(maxfev):
                    data[j, k] = np.where(conv[:, j] <= k)[0].size
                data[j, maxfev:] = data[j, maxfev - 1]
            data /= len(self.problems)

            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111)
            ax.yaxis.tick_left()
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(direction='in', which='both')
            for j, solver in enumerate(solvers):
                x = np.sort(perf[:, j])
                y = np.linspace(1 / len(self.problems), 1, len(self.problems))
                x = np.repeat(x, 2)[1:]
                y = np.repeat(y, 2)[:-1]
                x = np.r_[0, x[0], x, penalty * ratio_max]
                y = np.r_[0, 0, y, y[-1]]
                fmt = styles[j % len(styles)]
                plt.plot(x, y, fmt, label=solvers[j], linewidth=1)
            plt.xlim(0, 1.1 * max(1e-2, ratio_max))
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
                x = np.linspace(0, data_y / (self.n + 1), data_y)
                y = data[j, :]
                x = np.repeat(x, 2)[1:]
                y = np.repeat(y, 2)[:-1]
                fmt = styles[j % len(styles)]
                plt.plot(x, y, fmt, label=solvers[j], linewidth=1)
            plt.xlim(0, 1.1 * max(1e-2, ratio_max))
            plt.ylim(0, 1.1)
            plt.xlabel(r'$\mathrm{NF}/(n+1)$')
            plt.ylabel(fr'Data profile ($\tau=10^{{-{prec}}}$)')
            plt.legend(loc='lower right')
            pdf_data.savefig(fig, bbox_inches='tight')
            plt.close()
        pdf_perf.close()
        pdf_data.close()

    def run_solvers(self, solvers, options=None, load=True):
        options = self.set_default_options(options)
        maxfev = options.get('maxfev')
        merits = np.empty((len(self.problems), len(solvers), maxfev))
        for i, problem in enumerate(self.problems):
            print(f'Solving {problem.name} ({i + 1}/{len(self.problems)})...')
            for j, solver in enumerate(solvers):
                print(f'{solver:>10}:', end=' ')
                rec_path = self.get_rec_path(problem, solver)
                if load and rec_path.is_file():
                    history = np.load(rec_path)  # noqa
                    nfev = np.argmax(np.isnan(history))
                    if nfev == 0:
                        nfev = maxfev
                    merit = np.min(history[:nfev])
                    print(f'merit = {merit:.4e},', end=' ')
                    print(f'nfev = {nfev} (loaded).')
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        res = self.minimize(problem, solver, options)
                    print(f'fun = {res.fun:.4e},', end=' ')
                    if hasattr(res, 'maxcv'):
                        print(f'maxcv = {res.maxcv:.4e},', end=' ')
                    print(f'nfev = {res.nfev}.')

                    nfev = min(res.nfev, maxfev)
                    history = np.full(maxfev, np.nan)
                    history[:nfev] = res.merits[:nfev]
                    np.save(rec_path, history)  # noqa
                history[nfev:] = history[nfev - 1]
                merits[i, j, :] = history
        return merits

    def minimize(self, problem, solver, options):
        merits = []
        if solver.lower() == 'cobyqa':
            options_ = dict(options)
            options_['debug'] = True
            res = cobyqa.minimize(lambda x: self.eval(x, problem, merits),
                                  problem.x0, xl=problem.xl, xu=problem.xu,
                                  Aub=problem.aub, bub=problem.bub,
                                  Aeq=problem.aeq, beq=problem.beq,
                                  cub=problem.cub, ceq=problem.ceq,
                                  options=options_)
            res.merits = merits
        elif solver.lower() in pdfo.__all__:
            options_ = dict(options)
            options_['eliminate_lin_eq'] = False
            bounds = pdfo.Bounds(problem.xl, problem.xu)
            kwargs = {
                'fun': self.eval,
                'x0': problem.x0,
                'args': (problem, merits),
                'bounds': bounds,
                'options': options_,
            }
            if solver.lower != 'pdfo':
                kwargs['method'] = solver.lower()
            constraints = []
            if problem.mlub > 0:
                c = pdfo.LinearConstraint(problem.aub, -np.inf, problem.bub)
                constraints.append(c)
            if problem.mleq > 0:
                c = pdfo.LinearConstraint(problem.aeq, problem.beq, problem.beq)
                constraints.append(c)
            if problem.mnlub > 0:
                dub = np.zeros(problem.mnlub)
                c = pdfo.NonlinearConstraint(problem.cub, -np.inf, dub)
                constraints.append(c)
            if problem.mnleq > 0:
                deq = np.zeros(problem.mnleq)
                c = pdfo.NonlinearConstraint(problem.ceq, deq, deq)
                constraints.append(c)
            if len(constraints) > 0:
                kwargs['constraints'] = constraints
            res = pdfo.pdfo(**kwargs)
            res.merits = merits
            if hasattr(res, 'constrviolation'):
                res.maxcv = res.constrviolation
                del res.constrviolation
        else:
            raise NotImplementedError
        return res

    @staticmethod
    def eval(x, problem, merits):
        fx = problem.fun(x)
        maxcv = problem.maxcv(x)
        nan = np.finfo(type(fx)).max
        if maxcv >= 1e-2:
            merits.append(nan)
        else:
            merits.append(np.nan_to_num(fx + 1e6 * maxcv, nan=nan))
        return fx
