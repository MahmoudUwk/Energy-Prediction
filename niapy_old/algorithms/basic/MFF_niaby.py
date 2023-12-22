import logging

import numpy as np
import random
from niapy.algorithms.algorithm import Algorithm
from niapy.util.distances import euclidean

__all__ = ['Mod_FireflyAlgorithm']

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')


class Mod_FireflyAlgorithm(Algorithm):
    r"""Implementation of Firefly algorithm.

    Algorithm:
        Firefly algorithm

    Date:
        2016

    Authors:
        Iztok Fister Jr, Iztok Fister and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Fister, I., Fister Jr, I., Yang, X. S., & Brest, J. (2013).
        A comprehensive review of firefly algorithms. Swarm and Evolutionary Computation, 13, 34-46.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        alpha (float): Randomness strength.
        beta0 (float): Attractiveness constant.
        gamma (float): Absorption coefficient.
        theta (float): Randomness reduction factor.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['Mod_FireflyAlgorithm', 'ModFA']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Fister, I., Fister Jr, I., Yang, X. S., & Brest, J. (2013). A comprehensive review of firefly algorithms. Swarm and Evolutionary Computation, 13, 34-46."""

    def __init__(self, population_size=20, alpha=1, beta0=1, gamma=0.01, theta=0.97, eta = 0.4,*args, **kwargs):
        """Initialize Mod_FireflyAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            alpha (Optional[float]): Randomness strength 0--1 (highly random).
            beta0 (Optional[float]): Attractiveness constant.
            gamma (Optional[float]): Absorption coefficient.
            theta (Optional[float]): Randomness reduction factor.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.theta = theta
        self.eta = eta

    def set_parameters(self, population_size=20, alpha=1, beta0=1, gamma=0.01, theta=0.97,eta = 0.4, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            alpha (Optional[float]): Randomness strength 0--1 (highly random).
            beta0 (Optional[float]): Attractiveness constant.
            gamma (Optional[float]): Absorption coefficient.
            theta (Optional[float]): Randomness reduction factor.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.theta = theta
        self.eta = eta

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        params = super().get_parameters()
        params.update({
            'alpha': self.alpha,
            'beta0': self.beta0,
            'gamma': self.gamma,
            'theta': self.theta,
            'eta':self.eta
        })
        return params
    
    def FindLimits(self, k,task):
        for i in range(self.D):
            if self.Fireflies[k][i] < task.lower:
                self.Fireflies[k][i] = task.lower
            if self.Fireflies[k][i] > task.upper:
                self.Fireflies[k][i] = task.upper

    def init_ffa(self,task):
        Fireflies = np.zeros((task.population_size,task.dimension))
        Fitness = np.zeros((task.population_size))
        I = np.zeros((task.population_size))
        for j in range(task.dimension):
            Fireflies[0][j] = random.uniform(0, 1) * (task.upper- task.lower) + task.lower #X0
        for i in range(1,task.population_size):
            Fireflies[i] = self.eta * Fireflies[i-1] * (1-Fireflies[i-1]) #logistic map
            self.FindLimits(i,task)
            Fitness[i] = 1.0  # initialize attractiveness
            I[i] = Fitness[i]
        return Fireflies,I
            
    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * alpha (float): Randomness strength.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        fireflies, intensity = self.init_ffa(self,task)
        # fireflies, intensity, _ = super().init_population(task)
        return fireflies, intensity, {'alpha': self.alpha}

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Firefly Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current population function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individual fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution
                4. New global best solutions fitness/objective value
                5. Additional arguments:
                    * alpha (float): Randomness strength.

        See Also:
            * :func:`niapy.algorithms.basic.FireflyAlgorithm.move_ffa`

        """
        alpha = params.pop('alpha') * self.theta

        for i in range(self.population_size):
            for j in range(self.population_size):
                if population_fitness[i] >= population_fitness[j]:
                    r = euclidean(population[i], population[j])
                    beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                    steps = alpha * (self.random(task.dimension) - 0.5) * task.range
                    population[i] += beta * (population[j] - population[i]) + steps
                    population[i] = task.repair(population[i])
                    population_fitness[i] = task.eval(population[i])
                    best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)

        return population, population_fitness, best_x, best_fitness, {'alpha': alpha}