# -*- coding: utf-8 -*-




"""Optimization class combining CMA-ES with LCB method."""

import cirq
from copy import copy, deepcopy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy
from cma import CMAEvolutionStrategy, CMAOptions  # type: ignore
from cma.evolution_strategy import _CMAParameters
from scipy.stats import norm


IntList = List[int]

# pylint: disable=dangerous-default-value
class Solution(object):
    def __init__(
        self,
        x: Sequence = None,
        sample_mean: float = None,
        N: int = 0,
        fun: Callable = None,
        fun_args: Dict = {},
    ):
        self.x = x
        self.samples_N: numpy.ndarray = numpy.empty(0, dtype="int")
        self.samples: numpy.ndarray = numpy.empty(0, dtype="float")
        self.fun = fun
        self.fun_args = fun_args

        # Roberto: before it said f self.sample_mean is None
        # Now I am going to change it 
        if not hasattr(self, 'sample_mean'):
            self.N = 0
            self.sample_mean = 0
        else:
            self.samples += [sample_mean]
            self.N += [N]

    def update(
        self,
        sample_mean: float,
        N: int,
    ):
        """Given a new cost value for a certain candidate, add it to the list of 
        means.
        """
        self.samples = numpy.r_[self.samples, sample_mean]
        self.samples_N = numpy.r_[self.samples_N, N]
        self.sample_mean = (self.sample_mean * self.N + sample_mean * N) / (self.N + N)
        self.N += N
        return self

    def evaluate_and_update(self, N: int):
        """Given a candidate, evaluate the cost function with the candidate's parameters.
        After, update the result for the candidate.
        """
        self.fun_args['simulator'] = cirq.Simulator()
        sample_mean_ = self.fun(parameters=self.x, nshots=N, **self.fun_args)
        if not numpy.isnan(sample_mean_):
            self.update(sample_mean_, N)
        return self

    def get_se(self):
        """Jackknife estimate of the standard error of the sample expectation
        For each batch of n shots, we obtained a sample expectation, which converges
        to a Gaussian according to CLT:
        \hat{e} → N(E, \sigma^2 / n)
        And the final expectation estimation is the average over all such batches, e.g.,
        \hat{E} = \sum_{i=1}^B \hat{e}_i / B (given B batches in total)
        """
        n_group = len(self.samples)
        if n_group == 1:
            # return NaN if there is only one batch of shots allocated
            # as the SE estimate is not defined in this case
            return numpy.nan

        # sse = 0
        # for i in range(n_group):
        #     idx = [True] * n_group
        #     idx[i] = False
        #     v = (self.samples[idx] * self.samples_N[idx]).sum() / self.samples_N[idx].sum()
        #     sse += (v - self.sample_mean) ** 2
        # return numpy.sqrt(sse * (n_group - 1) / n_group)
        return numpy.std(self.samples, ddof=1) / numpy.sqrt(n_group)

    def __str__(self):
        return f"x: {self.x}\n" f"expectation: {self.sample_mean}\n" f"shots: {self.N}"

    def __repr__(self):
        return self.__str__()


# pylint: disable=dangerous-default-value
class LCB_CMAES(CMAEvolutionStrategy):
    """Optimization method combining CMAES with LCB."""

    def __init__(
        self,
        fun: Callable,
        x0: Sequence,
        sigma0: float,
        min_nb_shots_candidate: int,
        increment_nb_shots: int,
        total_nb_shots_iteration: int,
        total_nb_shots: int,
        *,
        beta_threshold: float = 0.95,
        CMA_options: Optional[Dict] = CMAOptions(),
        fun_args: Dict = {},
        fun_noiseless: Callable = None,
        fraction_var_reduction: float = 0.3,
    ):
        """
        Initialize class.
        Args:
            fun: function to optimize.
            x0: starting parameters of optimization function.
            sigma0: starting sigma0 of CMAEvolutionStrategy.
            min_nb_shots_candidate: minimum number of shots assigned to
            increment_nb_shots: number of shots to add to a valid candidate.
            total_nb_shots_iteration: maximum number of shots allowed
                per iteration.
                Includes minimum number of shots per candidate and increament.
            total_nb_shots: maximum number of shots allowed for the whole
                optimization procedure.
            beta_threshold: confidence level to stop the iteration, unless
                shots per iteration reaches a its maximum.
            CMA_options: Dictionary of CMA options. If none, takes CMAOptions
                 by default.
            fun_args: Additional arguments to pass to the function if
                 necessary.
        """
        # Initialize function and args
        self.fun = fun
        self.fun_args = fun_args
        self.fun_noiseless = fun_noiseless

        if (
            min_nb_shots_candidate < 0
            or increment_nb_shots < 0
            or total_nb_shots_iteration < 0
            or total_nb_shots < 0
        ):
            raise ValueError("Shots must be a positive integer.")

        if (
            min_nb_shots_candidate > total_nb_shots
            or min_nb_shots_candidate > total_nb_shots_iteration
            or total_nb_shots_iteration > total_nb_shots
        ):
            raise ValueError("Number of shots is not compatible.")

        # Start class variables related to shots
        self.min_nb_shots_candidate = min_nb_shots_candidate
        self.increment_nb_shots = increment_nb_shots
        self.total_nb_shots_iteration = total_nb_shots_iteration
        self.total_nb_shots = total_nb_shots
        self._total_nb_shots_optimization = int((1 - fraction_var_reduction) * total_nb_shots)
        
        # Roberto: I add this to keep account of the info for the plotting
        self.best_expectations_noiseless = []
        self.best_expectations_noisy = []
        self.used_shots_per_iteration = []

        if beta_threshold > 1.0 or beta_threshold < 0.0:
            raise ValueError("Beta threshold must be between 0 and 1.")

        # class variables related to LCB
        self._beta_threshold = beta_threshold
        self._norm_H_1 = numpy.sum(numpy.abs(self.fun_args["hamiltonian_terms"]))
        self._H_cross_term = numpy.sqrt(
            numpy.sum(
                numpy.abs(
                    numpy.outer(
                        self.fun_args["hamiltonian_terms"], self.fun_args["hamiltonian_terms"]
                    )
                )
            )
        )

        if "CMA_mu" in CMA_options:
            CMA_options["CMA_mu"] = int(
                numpy.floor(CMA_options["popsize"] * CMA_options["CMA_mu"])
            )

        # Initialize CMA inhereted class
        super().__init__(x0=x0, sigma0=sigma0, inopts=CMA_options)

        # the history of expectations values
        self.hist_true_expectations: List[float] = []
        # the history of estimated expectation values
        self.hist_expectations: List[float] = []
        # the number of shots consumed when observing the expectation value
        self.hist_shots: IntList = []
        # the number of shots allocated to each candidate in each iteration
        self.hist_shots_per_iteration: List[IntList] = []

        # the current number of shots consumed per each iteration of CMA-ES
        self.current_shots_used: int = 0
        self.kcma: int = None  # the number of candidates selected in LCB

        self._best: Solution = Solution()  # the best-so-far in terms of expectation only
        self._final: Solution = None  # the final recommendation (Solution object)
        self.final_x: List[float] = []  # the final point
        # self.final_expectation: float = None  # its expectation
        # self.final_se: float = None  # its standard error
        # self.final_shots: int = None  # its shots allocated
        self.pool: List[Solution] = []  # a pool of elite candidates

    @staticmethod
    def _get_beta(t: int = 0) -> float:
        """
        Get confidence level beta based on iteration.
        Args:
            t: iteration counter.
        Returns:
            confidence level.
        """
        if t < 0:
            raise ValueError("Iteration must be a positive integer.")

        while True:
            yield 1 - 1 / (t + numpy.sqrt(2)) ** 2
            t += 1

    # Here the shots are given to calculate the LCB? Or they are just given to calculate the expectation and then used for this formula?
    def _compute_LCB(self, expectation_value: float, beta: float, nb_shots: int) -> float:
        r"""
        Compute the lower-confidence bound.
        ..math::
            LCB(M_{0}, \theta) = \langle H \rangle(\theta) - epsilon *
                norm_H_1 / sqrt(4 M_{0}).
        Args:
            expectation_value: measured energy of a candidate.
            beta: confidence level.
            nb_shots: number of measurements done to estimate
                expectation value.
        Returns:
            lower-confidence bound value.
        """
        return expectation_value - norm.ppf(beta) * numpy.sqrt(
            self._norm_H_1 * self._H_cross_term / nb_shots
        )

    def function_eval(self, parameters: Sequence, nshots: int) -> float:
        """
        Call to a circuit expectation evaluation.
        Args:
            parameters: list of paramters for the circuit.
            nshots: current number of shots for candidate.
        Return:
            energy expectation value.
        """
        return self.fun(parameters=parameters, nshots=nshots, **self.fun_args)

    def _register_LCB_info(self, hist: List[Tuple[int, int]], candidates: List[Solution]):
        N = len(candidates)
        LCB_idx = [_[0] for _ in hist]
        LCB_shots = [_[1] for _ in hist]
        
        # Roberto: I put this here because it was giving an error some times
        try:
            last_idx = [numpy.nonzero(numpy.array(LCB_idx) == i)[0][-1] for i in range(N)]
            idx = numpy.argsort(last_idx)
            last_idx = [last_idx[_] for _ in idx]
    
            self.hist_expectations += [candidates[_].sample_mean for _ in idx]
            self.hist_shots += [LCB_shots[i] for i in last_idx]
            self.hist_shots_per_iteration.append([_.N for _ in candidates])
        except:
            print('There was an error in indices while doing the var reduction phase')
            
        
    def get_info_for_plotting(self):
        """
        Register the results of the best candidates after each iteration
        Args:
            self: only the optimiser object (self) is passed to this function
        Return:
            None
        """
        self.best_expectations_noisy.append(self._best.sample_mean)
        
        self.fun_args['simulator'] = cirq.Simulator()
        noiseless_best = self.fun_noiseless(parameters=self._best.x, **self.fun_args)
        self.best_expectations_noiseless.append(noiseless_best)
        
        print(f'The best noisy value is {self._best.sample_mean} and the best noiseless value is {noiseless_best}')
        
        

    def _evaluate_noiseless(self, candidates):
        if self.fun_noiseless is not None:
            self.hist_true_expectations += [
                self.fun_noiseless(parameters=x, **self.fun_args) for x in candidates
            ]

    def _LCB_evaluation(self, candidates: List[Solution], nshots: int = None) -> List:
        """Allocate the shots (re-evaluations) among ``candidates`` using the Lower
        Confidence Bound (LCB) method
        Parameters
        ----------
        candidates : List[Solution]
            list of parameter candidates to evaluate.
        nshots: int, optional
            the number of shots allowed for in this function call, by default None, which
            results in the smaller of ``self.total_nb_shots_iteration`` and the number of
            remaining shots of the entire CMA-ES
        Returns
        -------
        List
            a list of expectations
        Raises
        ------
        ValueError
            raise a value error when the number of candidates larger than popsize.
        """
                
        if len(candidates) > self.sp.popsize:
            raise ValueError("Number of candidates larger than popsize.")

        if nshots is None:
            shots_in_iteration = min(
                self.total_nb_shots_iteration,
                self._total_nb_shots_optimization - self.current_shots_used,
            )
        else:
            shots_in_iteration = nshots

        _hist = []  # list of (index of selected candidate, #shots used so far)
        for i, _ in enumerate(candidates):
            if _.N == 0:  # only evaluate candidates without any shots
                _.evaluate_and_update(self.min_nb_shots_candidate)
                self.current_shots_used += self.min_nb_shots_candidate
                _hist += [(i, self.current_shots_used)]
                shots_in_iteration -= self.min_nb_shots_candidate

        beta = LCB_CMAES._get_beta(0)
        while shots_in_iteration >= self.kcma * self.increment_nb_shots:
            # Update the LCB values for all candidate
            _beta = next(beta)
            _LCB_values = [self._compute_LCB(_.sample_mean, _beta, _.N) for _ in candidates]

            # Sort the indices based on LCB up to the candidates to pass.
            lcb_idx = numpy.argsort(_LCB_values)[0 : self.kcma]

            # Increase shots to lcb selected candidates.
            for idx in lcb_idx:
                candidates[idx].evaluate_and_update(self.increment_nb_shots)
                self.current_shots_used += self.increment_nb_shots
                _hist += [(idx, self.current_shots_used)]
                shots_in_iteration -= self.increment_nb_shots

        # registration of info.
        self._register_LCB_info(_hist, candidates)
        return [_.sample_mean for _ in candidates]

    # This method isn't called once in the whole code
    @staticmethod
    def get_default_kcma(dim: int, inopts: dict) -> int:
        """The default value of `self.kcma` is `self.sp.mu`, i.e., the number of candidates
        selected from the population. Such a default value is reasonable because the mean of the
        sampling distribution is (re-)estimated only from the selected candidates. Hence, by
        identifying the elites from a population more accurately, the LCB method helps improve
        the precison of adapting distributions' mean.
        """
        # .complement is used to add all missing options with default values
        # .evalall evaluates all option values in invironment loc
        # ???? what does this mean?
        options = CMAOptions(inopts).complement().evalall({"N": dim})
        if options["CMA_mu"] < 1:
            options["CMA_mu"] = int(numpy.floor(options["popsize"] * options["CMA_mu"]))
        # This returns the _CMAParameters, which are strategy parameters (sp) similar to CMAOptions
        return int(_CMAParameters(dim, options, verbose=options["verbose"] > 0).mu)

    def run(self, kcma: int = None, var_reduction: bool = False):
        """
        Run the LCB-CMAES algorithm
        
        Parameters
        ----------
        kcma : int, optional
            Parameter used for the CMAES algorithm. The default is None.
        var_reduction : bool, optional
            This boolean indicates if a portion of the total shots are maintained
            until the end of the optimisation to do a final variance reduction phase.
            The default is False.

        Returns
        -------
        None.

        """
        self._initialize(kcma)
        self._optimize()
        
        final_elites = self.pool
        if var_reduction:
            print(
                "Exploration phase finished."
                f"Variance reduction phase with {self.shots_left} shots."
            )
            self._var_reduction()
            # NOTE: `final_elites` is already sorted from the best to the worst
            final_elites = self._recommend(candidates=self.pool, k=self.sp.mu)

        # NOTE: `final_elites` -> the elites used to calculate `final_x`
        # `final_elites_shots` -> the shots allocated to each elite
        # `final_elites_expectations` -> the estimated energy of easch elite
        # print('final elites are:', final_elites)
        self.final_elites = [ind.x for ind in final_elites]
        self.final_elites_shots = numpy.asarray([ind.N for ind in final_elites])
        self.final_elites_expectations = numpy.asarray([ind.sample_mean for ind in final_elites])

        # NOTE: the final solution is a weighted combination of selected candidates
        weights = numpy.asarray(self.sp.weights.positive_weights)
        # print('self final elites are:', self.final_elites)
        # np.atleast_2d makes the input a two dimension array or more. So if the input is 2, the output is [[2]]
        self.final_x = weights.dot(numpy.atleast_2d(self.final_elites)[0 : len(weights), :])
        print(f'The best solution is {self._best.x}')
        #print(f'The elites are {self.final_elites}')
        #print(f'The expectations of the elites are {self.final_elites_expectations}')
        #print(f"Best recommended candidate is {self.final_x}")

    def _initialize(self, kcma):
        if kcma:
            if not isinstance(kcma, int):
                raise TypeError("Number of candidates to pass to CMA must be integer.")
            if kcma > self.opts["popsize"] or kcma < 0:
                raise ValueError("Number of candidates to CMA must be as large as popsize.")
            self.kcma = kcma
        else:
            # self.sp is: <cma.evolution_strategy._CMAParameters object at 0x7fddb47aeb20>
            # self.sp must be assigned when the superclass is instantiated
            self.kcma = self.sp.mu

    def _optimize(self):
        """
        Run a combined optimized of CMA-ES and LCB method.
        Returns:
            CMAEvolutionaryStrategy with the LCB best value.
        """
        counter = 0
        while True:
            shots_left = int(self._total_nb_shots_optimization - self.current_shots_used)
            if shots_left < self.sp.popsize * self.min_nb_shots_candidate:
                break

            print(f"Iteration number {counter} with shots used {self.current_shots_used}")
            # Get candidates from CMA-ES
            candidates = self.ask()

            # function evaluation: we use the LCB method to handle the noise
            _candidates = [Solution(x=x, fun=self.fun, fun_args=self.fun_args) for x in candidates]
            expectations = self._LCB_evaluation(_candidates)
            counter += 1

            # Update CMA parameters
            # NOTE: `CMA_cmean` is taken care of within `tell`
            self.tell(candidates, expectations)

            # maintain `xfavorite` manually, which is the mean of the sampling distribution
            # NOTE: `xfavorite_elites` -> the elites used to calculate `xfavorite`
            # `xfavorite_elites_shots` -> the shots allocated to each elite
            idx = numpy.argsort(expectations)[: self.sp.mu]
            self.xfavorite = self.mean.copy()
            self.xfavorite_elites = [_candidates[i].x for i in idx]
            self.xfavorite_elites_shots = numpy.asarray([_candidates[i].N for i in idx])
            self.xfavorite_elites_expectations = numpy.asarray(
                [_candidates[i].sample_mean for i in idx]
            )

            # Maintain 1) a pool of elite candidates and 2) ``self._best``
            # `self._best` is the single best-so-far (in terms of noisy energy) candidate
            # we ever sampled, which is only used for logging.
            self._maintain_candidate_pool(_candidates)
            
            # Roberto: I add this to keep account of the used shots
            self.used_shots_per_iteration.append(self.current_shots_used)
            
            # Roberto: here I define 50 as the number of shots to calculate the expectation of the candidates.
            # TODO: see the real value with the changes that Xavi does, e.g. when some shots are used to improve the elites
            self.get_info_for_plotting()

            print(
                f"Best-so-far: {self._best.sample_mean:.6f} +/- {self._best.get_se():.8f}"
                f" with {self._best.N} shots.\n"
            )
        self.shots_left = self.total_nb_shots - self.current_shots_used
        print('Number of shots left are', self.shots_left)
        return self

    @staticmethod
    def _recommend(candidates: List[Solution], k: int = 1):
        """select a subset of `k` solutions from `candidates`, considering the estimated energy and
        standard error of each solution
        """
        expectations = numpy.array([_.sample_mean for _ in candidates])
        se = numpy.array([_.get_se() for _ in candidates])
        # NOTE: '-1' is necessary since we are minimizing
        ratio = se / (-1 * expectations)

        idx = numpy.argsort(ratio)
        # the standard error will be NaN if only one batch of shots is allocated to
        # a candidate and in this case we rank such candidates using their expectations
        idx_nan = numpy.nonzero(numpy.isnan(ratio))[0]
        _n = len(ratio) - len(idx_nan)
        if _n < k:
            _ = numpy.argsort(expectations[idx_nan])[: (k - _n)]
            idx = numpy.r_[idx[:_n], idx_nan[_]]
        else:
            idx = idx[:k]
        return candidates[idx[0]] if k == 1 else [candidates[i] for i in idx]

    def _var_reduction(self, nshots: int = None):
        """Reduce the variance of the final candidate solutions
        Parameters
        ----------
        nshots : int
            the total number of shots allowed for this stage
        """
        if nshots is None:
            nshots = self.total_nb_shots - self.current_shots_used

        contain__best_so_far = any(
            [numpy.all(numpy.isclose(self._best.x, _.x)) for _ in self.pool]
        )
        # add ``self._best`` to the pool if not in and kick one out
        if not contain__best_so_far:
            # Roberto: here I do the append first and then input self.pool to self._recommend
            # Before the append went directly into self._recommend
            self.pool.append(self._best)
            self.pool = self._recommend(self.pool, self.sp.popsize)

        self._LCB_evaluation(self.pool, nshots)
        self.used_shots_per_iteration.append(self.total_nb_shots)
        self.get_info_for_plotting()
        return self

    def _maintain_candidate_pool(self, candidates: List[Solution]):
        """Maintain a pool of promising candidate solutions (of size `popsize`)
        It merges the new candidates (`candidates`) with the current pool (`self.pool`) and selects
        `popsize` elites from it using `self._recommend`.
        """
        
        # The type of candidates is List[Solution]
        if len(self.pool) == 0:
            # Elimino fun_args['simulator'] porque da un error TypeError: cannot pickle 'module' object
            for candidate in candidates:
                candidate.fun_args.pop('simulator', None)
                
            self.pool = deepcopy(candidates)
            for candidate in self.pool:
                candidate.fun_args['simulator'] = cirq.Simulator()
            print('The candidate\'s simulator is', self.pool[0].fun_args['simulator'])
        else:
            self.pool = self._recommend(self.pool + candidates, self.sp.popsize)

        # maintain ``self._best``
        idx = numpy.argmin([_.sample_mean for _ in candidates])
        _curr_best = candidates[idx].sample_mean
        if self._best.x is None or _curr_best < self._best.sample_mean:
            self._best = copy(candidates[idx])
        return self