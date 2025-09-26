from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import datetime
import math
import pickle
import random
import signal
import sys
import time


def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))


def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)   # get hours and remainder
    m, s = divmod(s, 60)     # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)


class Annealer(object):

    """Performs simulated annealing by calling functions to calculate
    energy and make moves on a state.  The temperature schedule for
    annealing may be provided manually or estimated automatically.
    """

    __metaclass__ = abc.ABCMeta

    # defaults
    Tmax = 25000.0
    Tmin = 2.5
    steps = 50000
    updates = 100
    copy_strategy = 'deepcopy'
    user_exit = False
    save_state_on_exit = False

    # placeholders
    best_state = None
    best_energy = None
    start = None

    def __init__(self, initial_state=None, load_state=None):
        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        elif load_state:
            self.load_state(load_state)
        else:
            raise ValueError('No valid values supplied for neither \
            initial_state nor load_state')

        signal.signal(signal.SIGINT, self.set_user_exit)

    def save_state(self, fname=None):
        """Saves state to pickle"""
        if not fname:
            date = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
            fname = date + "_energy_" + str(self.energy()) + ".state"
        with open(fname, "wb") as fh:
            pickle.dump(self.state, fh)

    def load_state(self, fname=None):
        """Loads state from pickle"""
        with open(fname, 'rb') as fh:
            self.state = pickle.load(fh)

    @abc.abstractmethod
    def move(self):
        """Create a state change"""
        pass

    @abc.abstractmethod
    def energy(self):
        """Calculate state's energy"""
        pass

    def set_user_exit(self, signum, frame):
        """Raises the user_exit flag, further iterations are stopped
        """
        self.user_exit = True

    def set_schedule(self, schedule):
        """Takes the output from `auto` and sets the attributes
        """
        self.Tmax = schedule['tmax']
        self.Tmin = schedule['tmin']
        self.steps = int(schedule['steps'])
        self.updates = int(schedule['updates'])

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of

        * deepcopy: use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()
        else:
            raise RuntimeError('No implementation found for ' +
                               'the self.copy_strategy "%s"' %
                               self.copy_strategy)

    def update(self, *args, **kwargs):
        """Wrapper for internal update.

        If you override the self.update method,
        you can chose to call the self.default_update method
        from your own Annealer.
        """
        self.default_update(*args, **kwargs)

    def default_update(self, step, T, E, acceptance, improvement):
        """Default update, outputs to stderr.

        Prints the current temperature, energy, acceptance rate,
        improvement rate, elapsed time, and remaining time.

        The acceptance rate indicates the percentage of moves since the last
        update that were accepted by the Metropolis algorithm.  It includes
        moves that decreased the energy, moves that left the energy
        unchanged, and moves that increased the energy yet were reached by
        thermal excitation.

        The improvement rate indicates the percentage of moves since the
        last update that strictly decreased the energy.  At high
        temperatures it will include both moves that improved the overall
        state and moves that simply undid previously accepted moves that
        increased the energy by thermal excititation.  At low temperatures
        it will tend toward zero as the moves that can decrease the energy
        are exhausted and moves that would increase the energy are no longer
        thermally accessible."""

        elapsed = time.time() - self.start
        if step == 0:
            print('\n Temperature        Energy    Accept   Improve     Elapsed   Remaining',
                  file=sys.stderr)
            print('\r{Temp:12.5f}  {Energy:12.2f}                      {Elapsed:s}            '
                  .format(Temp=T,
                          Energy=E,
                          Elapsed=time_string(elapsed)),
                  file=sys.stderr, end="")
            sys.stderr.flush()
        else:
            remain = (self.steps - step) * (elapsed / step)
            print('\r{Temp:12.5f}  {Energy:12.2f}   {Accept:7.2%}   {Improve:7.2%}  {Elapsed:s}  {Remaining:s}'
                  .format(Temp=T,
                          Energy=E,
                          Accept=acceptance,
                          Improve=improvement,
                          Elapsed=time_string(elapsed),
                          Remaining=time_string(remain)),
                  file=sys.stderr, end="")
            sys.stderr.flush()

    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials = accepts = improves = 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # Attempt moves to new states
        while step < self.steps and not self.user_exit:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials = accepts = improves = 0

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        # Return best state and energy
        return self.best_state, self.best_energy

    def auto(self, minutes, steps=2000):
        """Explores the annealing landscape and
        estimates optimal temperature settings.

        Returns a dictionary suitable for the `set_schedule` method.
        """

        def run(T, steps):
            """Anneals a system at constant temperature and returns the state,
            energy, rate of acceptance, and rate of improvement."""
            E = self.energy()
            prevState = self.copy_state(self.state)
            prevEnergy = E
            accepts, improves = 0, 0
            for _ in range(steps):
                dE = self.move()
                if dE is None:
                    E = self.energy()
                    dE = E - prevEnergy
                else:
                    E = prevEnergy + dE
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    self.state = self.copy_state(prevState)
                    E = prevEnergy
                else:
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prevState = self.copy_state(self.state)
                    prevEnergy = E
            return E, float(accepts) / steps, float(improves) / steps

        step = 0
        self.start = time.time()

        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature
        T = 0.0
        E = self.energy()
        self.update(step, T, E, None, None)
        while T == 0.0:
            step += 1
            dE = self.move()
            if dE is None:
                dE = self.energy() - E
            T = abs(dE)

        # Search for Tmax - a temperature that gives 98% acceptance
        E, acceptance, improvement = run(T, steps)

        step += steps
        while acceptance > 0.98:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        while acceptance < 0.98:
            T = round_figures(T * 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmax = T

        # Search for Tmin - a temperature that gives 0% improvement
        while improvement > 0.0:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmin = T

        # Calculate anneal duration
        elapsed = time.time() - self.start
        duration = round_figures(int(60.0 * minutes * step / elapsed), 2)

        # Don't perform anneal, just return params
        return {'tmax': Tmax, 'tmin': Tmin, 'steps': duration, 'updates': self.updates}


# Lightweight wrapper for discrete simulated annealing selection
# This complements the generic Annealer above with a simple API used
# across the project for picking an item that minimizes an energy fn.
class SimulatedAnnealing:
    """Discrete Simulated Annealing helper.

    Minimizes a provided energy function over a finite set of items.

    Parameters
    - initial_temperature: starting temperature for the schedule
    - cooling_rate: multiplicative decay per call (0 < r < 1)
    - min_temperature: lower bound for temperature
    - steps_per_call: number of annealing steps per select() call
    - neighbourhood: one of {"adjacent", "random"} to pick candidates
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.99,
        min_temperature: float = 1e-3,
        steps_per_call: int = 20,
        neighbourhood: str = "random",
    ) -> None:
        if not (0.0 < cooling_rate < 1.0):
            raise ValueError("cooling_rate must be in (0, 1)")
        if steps_per_call <= 0:
            raise ValueError("steps_per_call must be positive")
        self.initial_temperature = float(initial_temperature)
        self.cooling_rate = float(cooling_rate)
        self.min_temperature = float(min_temperature)
        self.steps_per_call = int(steps_per_call)
        self.neighbourhood = neighbourhood
        self._temperature = float(initial_temperature)

    @property
    def temperature(self) -> float:
        return self._temperature

    def _next_temperature(self) -> float:
        # Decay temperature per call
        decayed = self._temperature * (self.cooling_rate ** max(1, self.steps_per_call))
        return max(self.min_temperature, decayed)

    def reset(self) -> None:
        self._temperature = float(self.initial_temperature)

    def select(self, items, energy_fn, start_index=None):
        """Select an item minimizing energy_fn using SA.

        Returns (best_item, best_energy).
        """
        items = list(items)
        if len(items) == 0:
            raise ValueError("items must be non-empty")
        if len(items) == 1:
            x = items[0]
            return x, float(energy_fn(x))

        n = len(items)
        if start_index is None or not (0 <= int(start_index) < n):
            start_index = random.randrange(n)
        start_index = int(start_index)

        neighbourhood_mode = (self.neighbourhood or "random").lower()

        # Local subclass of Annealer operating on an integer index state
        class _IndexAnnealer(Annealer):
            def __init__(self, initial_state: int):
                super().__init__(initial_state=initial_state)
                # Avoid noisy default update prints
                self.updates = 0

            def move(self):
                i = int(self.state)
                if neighbourhood_mode == "adjacent":
                    # Move to a neighbor index (wrap around)
                    # Randomly choose left or right neighbor
                    if n <= 2:
                        j = (i + 1) % n
                    else:
                        j = (i + (1 if random.random() < 0.5 else -1)) % n
                else:
                    # Jump to a random different index
                    j = random.randrange(n - 1)
                    if j >= i:
                        j += 1
                # Apply move; let annealer compute dE by re-evaluating energy
                self.state = j
                return None

            def energy(self):
                idx = int(self.state)
                return float(energy_fn(items[idx]))

        # Configure and run anneal for a short schedule window
        ann = _IndexAnnealer(initial_state=start_index)
        tmax = max(self.min_temperature, float(self._temperature))
        tmin = max(self.min_temperature, float(self._next_temperature()))
        # Guarantee tmax >= tmin for schedule stability
        if tmax < tmin:
            tmax, tmin = tmin, tmax
        ann.set_schedule({"tmax": tmax, "tmin": tmin, "steps": self.steps_per_call, "updates": 0})
        best_state, best_energy = ann.anneal()

        # Advance temperature for the next call
        self._temperature = self._next_temperature()

        return items[int(best_state)], float(best_energy)
