import os
import warnings
from pathlib import Path

import cobyqa
import numpy as np
import pdfo
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

BASE_DIR = Path(__file__).resolve(strict=True).parent
ARCH_DIR = Path(BASE_DIR, os.environ.get('PYCUTEST_CACHE'))
ARCH_DIR.mkdir(exist_ok=True)
from problem import CUTEstProblems  # noqa


class Profiles:

    def __init__(self, n, feature='plain', constraints='U', callback=None):
        self.n = n
        self.feature = feature
        self.constraints = constraints

        self.perf_dir = Path(ARCH_DIR, 'performance', str(self.n))
        self.data_dir = Path(ARCH_DIR, 'data', str(self.n))
        self.eval_dir = Path(ARCH_DIR, 'save', str(self.n))

        self.problems = CUTEstProblems(self.n, self.constraints, callback)
        print()
        print(f'*** {len(self.problems)} problem(s) loaded ***')
        print()

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    def __call__(self, solvers, options=None, load=True):
        solvers = sorted(solvers)
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

    def get_rec_path(self, solver):
        filename = f'hist-{solver.lower()}-{self.constraints}.nc'
        return Path(self.eval_dir, filename)

    def get_perf_path(self, solvers, prec):
        solvers = '_'.join(map(str.lower, solvers))
        filename = f'perf-{solvers}-{self.constraints}-{prec}.pdf'
        return Path(self.perf_dir, filename)

    def get_data_path(self, solvers, prec):
        solvers = '_'.join(map(str.lower, solvers))
        filename = f'data-{solvers}-{self.constraints}-{prec}.pdf'
        return Path(self.data_dir, filename)

    def profiles(self, solvers, options=None, load=True):
        options = self.set_default_options(options)
        merits = self.run_solvers(solvers, options, load)

        ratio_max = 1e-6
        penalty = 2
        dpi = 200
        maxfev = merits.shape[-1]
        data_y = int(1.2 * maxfev)
        line_styles = ['-', '--', '-.', ':']

        merits_min = np.min(merits, axis=(1, 2))
        f0 = merits[..., 0]
        for prec in range(1, 10):
            print(f'Creating plots with tau = 1e-{prec}.')
            tau = 10 ** (-prec)

            conv = np.full_like(f0, np.nan)
            for i, problem in enumerate(self.problems):
                for j, solver in enumerate(solvers):
                    tau_conv = tau * f0[i, j]
                    tau_conv += (1 - tau) * merits_min[i]
                    if np.min(merits[i, j, :]) <= tau_conv:
                        conv[i, j] = np.nanargmax(merits[i, j, :] <= tau_conv) + 1

            perf = np.full_like(f0, np.nan)
            for i, problem in enumerate(self.problems):
                if not np.all(np.isnan(conv[i, :])):
                    perf[i, :] = conv[i, :]
                    perf[i] /= np.nanmin(conv[i, :])
            perf = np.log2(perf)
            ratio_max = max(ratio_max, np.nanmax(perf))
            perf[np.isnan(perf)] = penalty * ratio_max

            data = np.full((len(solvers), data_y), np.nan)
            for j in range(len(solvers)):
                for k in range(maxfev):
                    data[j, k] = np.where(conv[:, j] <= k)[0].size / len(self.problems)
                data[j, maxfev:] = data[j, maxfev - 1]

            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111)
            ax.yaxis.tick_left()
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(direction='in', which='both')
            for j, solver in enumerate(solvers):
                x = np.repeat(np.sort(perf[:, j]), 2)[1:]
                y = np.repeat(np.linspace(1 / len(self.problems), 1, len(self.problems)), 2)[:-1]
                x = np.r_[0, x[0], x, penalty * ratio_max]
                y = np.r_[0, 0, y, y[-1]]
                plt.plot(x, y, line_styles[j % len(line_styles)], label=solvers[j], linewidth=1)
            plt.xlim(0, 1.1 * max(0.01, ratio_max))
            plt.ylim(0, 1.1)
            plt.xlabel(r'$\log_2(\mathrm{NF}/\mathrm{NF}_{\min})$')
            plt.ylabel(f'Performance profile ($\\tau=10^{{-{prec}}}$)')
            plt.legend(loc='lower right')
            plt.savefig(self.get_perf_path(solvers, prec), bbox_inches='tight')
            plt.close()

            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111)
            ax.yaxis.tick_left()
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(direction='in', which='both')
            for j, solver in enumerate(solvers):
                x = np.repeat(np.linspace(0, data_y / (self.n + 1), data_y), 2)[1:]
                y = np.repeat(data[j, :], 2)[:-1]
                plt.plot(x, y, line_styles[j % len(line_styles)], label=solvers[j], linewidth=1)
            plt.xlim(0, 1.1 * max(0.01, ratio_max))
            plt.ylim(0, 1.1)
            plt.xlabel(r'$\mathrm{NF}/(n+1)$')
            plt.ylabel(f'Data profile ($\\tau=10^{{-{prec}}}$)')
            plt.legend(loc='lower right')
            plt.savefig(self.get_data_path(solvers, prec), bbox_inches='tight')
            plt.close()

    def run_solvers(self, solvers, options=None, load=True):
        options = self.set_default_options(options)
        maxfev = options.get('maxfev')
        merits = np.empty((len(self.problems), len(solvers), maxfev))
        loaded = {}
        if load:
            pass
            # for solver in solvers:
            #     record_path = self.get_rec_path(solver)
            #     # TODO: Inclusions among the constraints are possible
            #     if record_path.is_file():
            #         loaded[solver] = xr.open_dataarray(record_path)
        n_problems = len(self.problems)
        for i, problem in enumerate(self.problems):
            print(f'Solving {problem.name} ({i + 1}/{n_problems})...')
            for j, solver in enumerate(solvers):
                print(f'{solver:>10}:', end=' ')
                if loaded.get(solver) is not None and problem.name in []:
                    # history = loaded.get(solver).loc[problem.name]
                    history = None
                    print('Loaded.')
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        res = self.minimize(problem, solver, options)
                    print(f'fun = {res.fun:.4e},', end=' ')
                    if hasattr(res, 'maxcv'):
                        print(f'maxcv = {res.maxcv:.4e},', end=' ')
                    print(f'nfev = {res.nfev}.')

                    nfev = min(res.nfev, maxfev)
                    history = np.empty(maxfev, dtype=float)
                    history[:nfev] = res.merits[:nfev]
                    history[nfev:] = res.merits[nfev - 1]
                merits[i, j, :] = history
        # for solver in da.coords['solver'].values:
        #     temp = da.loc[:, solver, :]
        #     if loaded.get(solver) is not None:
        #         temp = temp.combine_first(loaded.get(solver))
        #     temp.to_netcdf(self.get_rec_path(solver))
        return merits

    def minimize(self, problem, solver, options):
        merits = []
        if solver.lower() == 'cobyqa':
            options_ = dict(options)
            options_['debug'] = False  # FIXME
            res = cobyqa.minimize(lambda x: self.eval(x, problem, merits),
                                  problem.x0, xl=problem.xl, xu=problem.xu,
                                  Aub=problem.aub, bub=problem.bub,
                                  Aeq=problem.aeq, beq=problem.beq,
                                  cub=problem.cub, ceq=problem.ceq,
                                  options=options_)
            res.merits = merits
        elif solver.lower() in pdfo.__all__:
            options_ = dict(options)
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
                constraints.append(pdfo.LinearConstraint(problem.aub, -np.inf, problem.bub))
            if problem.mleq > 0:
                constraints.append(pdfo.LinearConstraint(problem.aeq, problem.beq, problem.beq))
            if problem.mnlub > 0:
                dub = np.zeros(problem.mnlub)
                constraints.append(pdfo.NonlinearConstraint(problem.cub, -np.inf, dub))
            if problem.mnleq > 0:
                deq = np.zeros(problem.mnleq)
                constraints.append(pdfo.NonlinearConstraint(problem.ceq, deq, deq))
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
        if maxcv >= 1e-2:
            merits.append(np.finfo(type(fx)).max)
        else:
            merits.append(fx + 1e3 * maxcv)
        return fx
