"""The core module contains the MSOModel class that runs the simulation."""

__docformat__ = 'reStructuredText'

import numpy as np
import os
import datetime
import json
import gzip
import sys
import shutil
import multiprocessing
import warnings
import getopt
import time

try:
    from memory_profiler import memory_usage
    mem_profiler_not_available = False
except ImportError:
    warnings.warn('Could not import memory_profiler, some profile functions may not work')
    mem_profiler_not_available = True


class MSOModel(object):
    """This class contains the model of the MSO and methods to run simulations

    The MSOModel is the main class that runs the simulation. See README.txt for instructions how to run it.

    :param timesteps: The number of timesteps (in milliseconds) the simulation should run
    :param dt: Stepsize of simulation (in ms)
    :param vt: Threshold for voltage (theta v)
    :param batchsize: The size of each batch (in milliseconds). For varying ITDs the batchsize determines how often the ITDs are changed.
    :param input_neurons: Number of input neurons (layer 1)
    :param MSO_neurons: Number of MSO neurons (layer 2)
    :param T: The period of the input sound (i.e. determines the frequency)
    :param storeData: Boolean that indicates whether or not simulation results should be stored
    :param ITDoptions: An array of ITDs the simulation can choose from (set to e.g. [0.0] for constant ITD of 0.0)
    :param path: Used for reloading model in a certain with :func:`MSOModel.loadLearnedModel()`, don't use yourself
    :param delays: Used for reloading models with :func:`MSOModel.loadLearnedModel()`, don't use yourself
    :param memorySavingMode: Determines the mode for memory saving. See note.

    .. note::
        Do not use memorySavingMode 0 or 1. These are only included for completeness (see appendix A). They are slower and take up more memory than memorySavingMode 2.
    :type timesteps: int
    :type batchsize: int
    :type input_neurons: int
    :type MSO_neurons: int
    :type T: float
    :type storeData: boolean
    :type ITDoptions: array or int
    :type dt: float
    :type vt: float
    :type path: string
    :type delays: numpy array
    :type memorySavingMode: 0, 1 or 2


    """

    # LIF parameters
    TAU_M = 0.100       # (ms)
    TAU_S = 0.100       # (ms)

    # Delay parameters
    DELAY_MU = 7.0      # (ms)
    DELAY_SIGMA = 1.0   # 0.300 # (ms)

    # Learning parameters
    TAU_LTP = 10.0      # (ms)
    TAU_LTD = 10.0      # (ms)
    A_LTP = 0.00100
    A_LTD = 0.00105

    SATURATION = 3.0    # Maximum weight
    REFRACTORY_PERIOD = 0.500

    ERROR = 0.00001     # Minimum value for mask of I and pre-trace

    STEP_DEGREES = 5.0  # Degrees (not radian)
    HEADSIZE = 0.1      # Meters
    SOUNDSPEED = 343.0  # Meters / second

    if A_LTD <= A_LTP:
        raise ValueError('For balanced learning, A_LTD ({}) must be higher than A_LTP ({})'.format(A_LTP, A_LTD))

    def __init__(self, timesteps, batchsize, vt, input_neurons, MSO_neurons, dt, T, storeData=True,
                 ITDoptions=[0.0], memorySavingMode=2, parallel=1, path=False, delays=False):

        c = self.__class__    # Shortcut to class variables

        # Check validity of input values
        if timesteps % batchsize != 0:
            raise ValueError('batchsize ({}) has to be a multiple of the given number of timesteps ({})'.format(batchsize, timesteps))
        if input_neurons % 2 != 0:
            raise ValueError('input_neurons ({}) must be an even number '.format(input_neurons))
        if (batchsize / dt) % (T / dt) > 0.000001:
            raise ValueError('batchsize ({}) has to be a multiple of the signal period (T = {})'.format(batchsize, T))
        if batchsize < np.ceil(-np.log(c.ERROR/c.A_LTP)*c.TAU_LTP):
            raise ValueError('Batchsize must be at least {}'.format(np.ceil(-np.log(c.ERROR/c.A_LTP)*c.TAU_LTP)))
        if parallel < 1:
            raise ValueError('Parallel ({}) must be at least 1'.format(parallel))
        if memorySavingMode not in [0, 1, 2]:
            raise ValueError('memorySavingMode ({}) must be either 0, 1, or 2'.format(batchsize, timesteps))
        if memorySavingMode in [0, 1]:
            warnings.warn('MemorySavingMode 0 and 1 are depricated, please use memeroySavingMode 2')

        # Set simulation parameters
        self.timesteps = timesteps
        self.batchsize = batchsize
        self.vt = vt
        self.input_neurons = input_neurons
        self.MSO_neurons = MSO_neurons
        self.dt = dt
        self.storeData = storeData
        self.ITD = 0.0
        self.T = T
        self.ITDoptions = ITDoptions
        self.memorySavingMode = memorySavingMode
        self.parallel = parallel

        # Directory where results will be stored
        if storeData:
            if not path:
                print "\tCreating directory..."
                self.path = self.createDirectory()
                self.storeParameters()
            else:
                self.path = self.createDirectory(path)
                self.storeParameters()
            print "Results will be stored in: {}".format(self.path)

        # Update formulas
        self.dv = lambda v, I: -v/c.TAU_M + I
        self.dltp = lambda x: -x/c.TAU_LTP
        self.dltd = lambda y: -y/c.TAU_LTD

        print "\tAllocating data..."

        # If no delays are included (i.e. model is not preloaded), generate delays
        if delays is False:
            # Generate delays and round them to dt
            self.delays = (np.random.normal(c.DELAY_MU, c.DELAY_SIGMA, (MSO_neurons, input_neurons))/dt).astype(int)*dt

            if storeData:
                with open('{}/delays.txt'.format(self.path), 'w') as outfile:
                    json.dump(self.delays.tolist(), outfile)
        else:
            self.delays = delays

        # Transform delays into number of timesteps rather than milliseconds
        self.delays_dt = (self.delays/dt).astype(int)
        self.maxDelay = int(np.max(self.delays_dt))

        # Transforms batchsize in number of timesteps rather than milliseconds
        self.batchsize_dt = int(batchsize/dt)
        self.t_range = np.arange(0, 2*self.batchsize_dt, dt)

        # Allocate space for fire handling and precomputations (see appendix A)
        # MemorySavingMode 0:
        #     Precompute I and presynaptic trace values and store spikes in an m x n xbatchsize array
        # MemorySavingMode 1:
        #      Store spikes in an m x n x batchsize array
        # MemorySavingMode 2:
        #      Store spikes in an 2 x batchsize array, where the first and second array represent
        #      a pre-post synaptic spike pair that arrives at a particular time in batchsize
        if memorySavingMode == 0:
            Ifj = lambda t, tfj: (t >= tfj)*(1 / c.TAU_S)*np.exp(-(t - tfj) / c.TAU_S)
            I_cutoff = np.ceil((-np.log(c.ERROR/(1/c.TAU_S))*c.TAU_S)/dt)
            self.I_mask_overlay = int(max(I_cutoff, c.REFRACTORY_PERIOD/self.dt))
            self.I_mask = np.array([Ifj(t, 0) for t in self.t_range[:self.I_mask_overlay]])

            pretrace = lambda t, tfj: (t >= tfj)*c.A_LTP*np.exp(-(t - tfj) / c.TAU_LTP)
            self.pretrace_mask_overlay = int(np.ceil((-np.log(c.ERROR/c.A_LTP)*c.TAU_LTP)/dt))
            self.pretrace_mask = np.array([pretrace(t, 0) for t in self.t_range[:self.pretrace_mask_overlay]])

            self.delayed_firings = np.zeros((MSO_neurons, input_neurons, self.batchsize_dt + self.maxDelay))
            self.I = np.zeros((MSO_neurons, input_neurons,
                               self.maxDelay + self.batchsize_dt + self.I_mask_overlay))
            self.pretrace = np.zeros((MSO_neurons, input_neurons,
                                      self.maxDelay + self.batchsize_dt + self.pretrace_mask_overlay))

        elif memorySavingMode == 1:
            self.dI = lambda I: -I/c.TAU_S
            self.I = np.zeros((MSO_neurons, input_neurons))
            self.pretrace = np.zeros((MSO_neurons, input_neurons))
            self.delayed_firings = np.zeros((MSO_neurons, input_neurons, self.batchsize_dt + self.maxDelay))

        elif memorySavingMode == 2:
            self.dI = lambda I: -I/c.TAU_S
            self.I = np.zeros((MSO_neurons, input_neurons))
            self.pretrace = np.zeros((MSO_neurons, input_neurons))
            self.delayed_firings = [[[[] for i in range(self.batchsize_dt + self.maxDelay)],
                                    [[] for i in range(self.batchsize_dt + self.maxDelay)]]
                                    for x in range(self.parallel)]

        # Data allocation general simulation
        self.prevFirings = np.zeros((input_neurons, self.batchsize_dt))
        self.v = np.zeros((MSO_neurons))
        self.v_spikes = np.zeros((MSO_neurons, self.batchsize_dt + 1))
        self.last_spike = -(c.REFRACTORY_PERIOD/self.dt)*np.ones(MSO_neurons)
        self.w = np.ones((MSO_neurons, input_neurons))
        self.posttrace = np.zeros(MSO_neurons)
        self.resume = 0

        # Set P-values for left and right ear
        self.setFrequency(T)
        self.setITD(0.0)
        self.testMode = False

        if parallel > 1:
            warnings.warn('Parallel > 1 is not thoroughly tested, race conditions may result in unpredictable behaviour ')

        self.mso_idx = np.array_split(np.arange(self.MSO_neurons), self.parallel)

        print "Initialisation succeeded"

    @classmethod
    def loadLearnedModel(cls, folder, t):
        """Class method that can be used to load a pre-existing model and test or run the model.

        :param folder: Path to the folder where the model is stored
        :param t: Time identifier of the weight matrix that should be loaded, i.e. if the filename is '19999_w.json.gz' the t value is 19999

        .. note::
            The value for t is the time step, not the time in milliseconds. If you would like
            to reload a simulation after 500 ms for a simulation with dt = 0.005, you compute the correct weight matrix as follows:

                500/0.005*1000 = 100000000
        """

        path = folder
        t = int(t)

        # Load parameters
        with open('{}/parameters.txt'.format(path), 'r') as f:
            classParams = dict(json.loads(f.readlines()[0]))

        # Load simulation settings
        with open('{}/simulation_settings.txt'.format(path), 'r') as f:
            objParams = dict(json.loads(f.readlines()[0]))

        # Load delays
        with open('{}/delays.txt'.format(path), 'r') as f:
            delays = np.array(json.loads(f.readlines()[0]))

        # For depricated simulations that did not have the memorySavingMode
        objParams['memorySavingMode'] = 2

        sim = cls(timesteps=objParams['timesteps'], batchsize=objParams['batchsize'], vt=objParams['vt'],
                  input_neurons=objParams['input_neurons'], MSO_neurons=objParams['MSO_neurons'],
                  dt=objParams['dt'], storeData=False, path=folder, T=objParams['T'],
                  delays=delays, memorySavingMode=objParams['memorySavingMode'])

        # Set all class and object values of the loaded simulation to the new one
        for (key, value) in classParams.iteritems():
            setattr(cls, key, value)

        for (key, value) in objParams.iteritems():
            setattr(sim, key, value)

        # Set weights to the data of the latest timestep
        with gzip.open('{}/data_per_batch/{}_w.json.gz'.format(path, t), 'r') as f:
            sim.w = np.array(json.loads(f.readlines()[0]))

        # Set resume variable from which the simulation is supposed to continue
        sim.resume = int((t + 1) / sim.batchsize_dt)

        return sim

    def computeITD(self, angle):
        """
            Computes ITD for a given angle (in degrees)
        """
        angle = np.radians(angle)
        return (self.__class__.HEADSIZE/self.__class__.SOUNDSPEED)*1000*(np.sin(angle) + angle)

    def setFrequency(self, T):
        """
            Resets P-values for the given period T of the input sound
        """
        self.T = T           # (ms)
        self.sigma = 0.08 + T*0.07     # T*0.2
        self.MFR = 0.150     # 1.0/T*0.2  # milliHertz
        self.mrange = 100

        # Generate p_values
        st_range = self.t_range[:int(round(self.T / self.dt))]
        p_val = np.array([self.P(t, 0.0) for t in st_range])

        self.p_values_left = np.tile(p_val, int(self.batchsize / self.T))
        self.p_values_right = np.tile(p_val, int(self.batchsize / self.T))

        # Reset to ITD
        if self.ITD:
            ITD = self.ITD
        else:
            ITD = 0.0
        self.ITD = 0.0
        self.setITD(ITD)

    def setITD(self, ITD):
        """
            Moves precomputed P-values such that a particular ITD is imposed
        """

        ITDdiff = self.ITD - ITD
        self.p_values_right[:] = np.roll(self.p_values_right, int(ITDdiff/self.dt))
        self.ITD = ITD

    def storeParameters(self):
        """
            Stores class and object parameters of this particular simulation
        """

        # Get all class variables
        variables = [attr for attr in dir(self.__class__) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        class_vars = dict([(var, vars(self.__class__)[var]) for var in variables])

        # Get all simulation variables
        obj_vars = vars(self)

        # Save delays
        with open('{}/parameters.txt'.format(self.path), 'w') as outfile:
            json.dump(class_vars, outfile)

        # Save delays
        with open('{}/simulation_settings.txt'.format(self.path), 'w') as outfile:
            json.dump(obj_vars, outfile)

    def createDirectory(self, path=False):
        """
            Creates directory structure in which the simulation is going to be stored
        """

        # Create file and directory
        if not path:
            directory = str(datetime.date.today())
            timestamp = str(datetime.datetime.now().time())[:-7].replace(':', '-')  # Give similar simulations a unique folder

            filename = '{}'.format(timestamp)
            path = '{}/{}'.format(directory, filename)

            if not os.path.exists(directory):
                os.makedirs(directory)

        else:
            shutil.rmtree(path)

        if not os.path.exists(path):
            os.makedirs(path)

        os.makedirs('{}/data_per_batch'.format(path))

        return path

    def P(self, t, delay):
        """
            Computes P-value for a particular time t (in ms) and delay (in ms)
        """
        return (self.MFR*self.T)/(self.sigma*np.sqrt(2*np.pi))*np.sum([np.exp(-(t - m*self.T - delay)**2 / (2*self.sigma**2))
                                                                       for m in range(-self.mrange, self.mrange)])

    def batchReset(self, firings, batch_I=False, batch_pretrace=False):
        """
            Resets simulation to prepare it for a new batch of firings (np.array).
            It precomputes all required values (it depends on the memorySavingMode which values are precomputed and which are not)
            and initiates **delayed_firings**, which contains all the produced spikes such that it is readable for the update function
        """
        if self.memorySavingMode == 0:
            # Reset I
            overlay = np.zeros(self.I.shape)
            overlay[:, :, :(self.I_mask_overlay + self.maxDelay)] = self.I[:, :, -(self.I_mask_overlay + self.maxDelay):]
            self.I[:] = overlay

            # Reset pretrace
            overlay = np.zeros(self.pretrace.shape)
            overlay[:, :, :(self.pretrace_mask_overlay + self.maxDelay)] = self.pretrace[:, :, -(self.pretrace_mask_overlay + self.maxDelay):]
            self.pretrace[:] = overlay

            # Reset delayed firings
            overlay = np.zeros(self.delayed_firings.shape)
            overlay[:, :, :self.maxDelay] = self.delayed_firings[:, :, -self.maxDelay:]
            self.delayed_firings[:] = overlay

            for neuron in range(self.MSO_neurons):
                for i in range(self.input_neurons):
                    self.delayed_firings[neuron, i, self.delays_dt[neuron, i]:(firings.shape[1] + self.delays_dt[neuron, i])] += firings[i]
                    self.I[neuron, i, self.delays_dt[neuron, i]:(batch_I.shape[1] + self.delays_dt[neuron, i])] += batch_I[i]
                    self.pretrace[neuron, i, self.delays_dt[neuron, i]:(batch_pretrace.shape[1] + self.delays_dt[neuron, i])] += batch_pretrace[i]
        elif self.memorySavingMode == 1:

            # Reset delayed firings
            overlay = np.zeros(self.delayed_firings.shape)
            overlay[:, :, :self.maxDelay] = self.delayed_firings[:, :, -self.maxDelay:]
            self.delayed_firings[:] = overlay

            for neuron in range(self.MSO_neurons):
                for i in range(self.input_neurons):
                    self.delayed_firings[neuron, i, self.delays_dt[neuron, i]:(firings.shape[1] + self.delays_dt[neuron, i])] += firings[i]

        elif self.memorySavingMode == 2:

            overlay = [(self.delayed_firings[x][0][-self.maxDelay:], self.delayed_firings[x][1][-self.maxDelay:]) for x in range(self.parallel)]
            timerange = range(self.batchsize_dt + self.maxDelay)
            self.delayed_firings = [([[] for i in timerange], [[] for i in timerange]) for x in range(self.parallel)]

            for x in range(self.parallel):
                self.delayed_firings[x][0][:self.maxDelay] = overlay[x][0]
                self.delayed_firings[x][1][:self.maxDelay] = overlay[x][1]

            (js, t_firings) = np.where(firings > 0)

            for x in range(self.parallel):
                for neuron in range(len(self.mso_idx[x])):
                    for i, t in enumerate(t_firings):
                        self.delayed_firings[x][0][t + self.delays_dt[neuron, js[i]]] += [neuron]
                        self.delayed_firings[x][1][t + self.delays_dt[neuron, js[i]]] += [js[i]]

        # Retrieve firings of the past s.t. a refractory period still holds when moving from one batch to another
        self.prevFirings = firings[:, -int(self.__class__.REFRACTORY_PERIOD / self.dt)]

    def generateAutoITD(self, nr_batches):
        """
            Method used to compute a sequence of ITDs according to the sound localisation task
        """
        step_degree = self.__class__.STEP_DEGREES
        theta_options = np.arange(-45, 45 + step_degree, step_degree)

        theta_head = np.random.choice(theta_options)
        theta_sound = np.random.choice(theta_options)
        thetas = np.zeros(nr_batches)
        thetas[0] = theta_head - theta_sound

        for i in range(1, nr_batches):
            theta_head += -np.sign(thetas[i - 1])*step_degree
            if thetas[i - 1] == 0:
                theta_sound = np.random.choice(theta_options)
            thetas[i] = theta_head - theta_sound

        return np.round(self.computeITD(thetas)/self.dt)*self.dt

    def run(self):
        """
            Core method that runs the simulation for the initialised parameters
        """
        nr_batches = int(((self.timesteps / self.dt)) / self.batchsize_dt)

        if self.ITDoptions == 1:
            # Use sound localisation task to generate input
            if self.resume == 0:
                ITDs = self.generateAutoITD(nr_batches)
            else:
                with gzip.open('{}/itds.json.gz'.format(self.path), 'r') as f:
                    ITDs = np.array(json.loads(f.readlines()[0]))
                print ITDs
        else:
            # Generate random sequence of the given set of ITDs
            ITDs = np.random.choice(self.ITDoptions, nr_batches)

        if self.storeData:
            with gzip.open('{}/itds.json.gz'.format(self.path), 'w') as outfile:
                json.dump(ITDs.tolist(), outfile)

        storeProcesses = []

        preTrain = 5
        if self.testMode:
            preTrain = 0

        for batch in range(self.resume - preTrain, nr_batches):

            print "Batch {}/{}".format(batch, nr_batches)
            if batch < self.resume:
                print "\tWarming up before learning resumes"
                learning = False
                self.setITD(0)
            else:
                learning = True
                print "\tITD selected: {}".format(ITDs[batch])
                self.setITD(ITDs[batch])

            offset = batch*self.batchsize_dt

            if self.memorySavingMode:    # memorySavingMode is 1 or 2
                firings = self.generateFirings()
                self.batchReset(firings)
            else:
                firings, batch_I, batch_pretrace = self.generateFirings()
                self.batchReset(firings, batch_I, batch_pretrace)

            if self.parallel > 1:
                processes = []

                queue = multiprocessing.Queue()
                processes = [multiprocessing.Process(target=self.runBatch, args=(queue, i, offset, learning)) for i in range(self.parallel)]
                for p in processes:
                    p.start()
                results = [queue.get() for i in range(self.parallel)]
                queue.close()
                queue.join_thread()

                for p in processes:
                    p.join()

                print "Storing Results"

                for x in range(self.parallel):
                    (i, (v, lastspike, v_spikes, I, pretrace, posttrace, w)) = results[i]
                    idx = self.mso_idx[i]
                    self.v[idx], self.last_spike[idx], self.v_spikes[idx] = v, lastspike, v_spikes
                    self.I[idx], self.pretrace[idx], self.posttrace[idx], self.w[idx] = I, pretrace, posttrace, w

            else:
                if self.memorySavingMode:
                    for t in range(self.batchsize_dt):
                        self.updateLowMemory(parallel_id=0, t=t, offset=offset, learning=learning)
                else:
                    for t in range(self.batchsize_dt):
                        self.update(parallel_id=0, t=t, offset=offset, learning=learning)

            if self.storeData and batch >= self.resume:
                print "\tStoring data..."
                p = multiprocessing.Process(target=self.storeBatch, args=(self.batchsize_dt - 1, batch, self.w.copy()))  # , self.v_spikes.copy()))
                p.start()
                storeProcesses.append(p)

        for p in storeProcesses:
            p.join()

    def runBatch(self, queue, i, offset, learning):
        """ 
            Performs update for a parallel process
        """
        idx = self.mso_idx[i]

        if self.memorySavingMode:
            for t in range(self.batchsize_dt):
                self.updateLowMemory(parallel_id=i, t=t, offset=offset, learning=learning)
        else:
            for t in range(self.batchsize_dt):
                self.update(parallel_id=i, t=t, offset=offset, learning=learning)

        data = self.v[idx], self.last_spike[idx], self.v_spikes[idx], self.I[idx], self.pretrace[idx], self.posttrace[idx], self.w[idx]
        queue.put((i, data))

    def storeBatch(self, t, batch, w, v_spikes=False):
        """
            Stores weights of timestep t
        """
        with gzip.open('{}/data_per_batch/{}_w.json.gz'.format(self.path, t + batch*self.batchsize_dt), 'w') as outfile:
            json.dump(w.tolist(), outfile)
        #with gzip.open('{}/data_per_batch/{}_vspikes.json.gz'.format(self.path, t + batch*self.batchsize_dt), 'w') as outfile:
        #    json.dump(v_spikes.tolist(), outfile)
        del w     # Garbage collection
        return

    def reset(self):
        """
            Used to reset all parameters, for example when the model is tested
        """
        self.prevFirings *= 0
        self.delayed_firings *= 0
        self.delayed_firings = [[[[] for i in range(self.batchsize_dt + self.maxDelay)],
                                [[] for i in range(self.batchsize_dt + self.maxDelay)]]
                                for x in range(self.parallel)]
        self.v *= 0
        self.v_spikes *= 0
        self.last_spike *= 0
        self.I *= 0
        self.pretrace *= 0
        self.posttrace *= 0

    def test(self, timesteps=False, ITDs=False, learning=False):
        """
            Similar as run(), but for testing purposes. Runs for the given number of timesteps (in ms)
            for the ITD it was set to (use :func:`setITD` for this purpose) or chooses a random
            ITD from the given set of ITDs. Returns all spikes in a [mso_neurons x (timesteps/dt)]
            array
        """
        self.reset()

        if timesteps and ITDs is not False:
            raise ValueError('The paramaters timesteps and ITDs cannot both be set to a value')

        if ITDs is not False:
            nr_batches = len(ITDs)
            timesteps_dt = nr_batches*self.batchsize_dt
        else:
            if not timesteps:
                timesteps = self.batchsize
            timesteps_dt = int(timesteps / self.dt)
            nr_batches = int(timesteps_dt / self.batchsize_dt)

        v_spikes = np.zeros((self.MSO_neurons, timesteps_dt))

        for batch in range(nr_batches):

            print "Batch {}/{}".format(batch, nr_batches)

            if ITDs is not False:
                print "\tITD selected:{}".format(ITDs[batch])
                self.setITD(ITDs[batch])

            if self.memorySavingMode:
                firings = self.generateFirings()
                self.batchReset(firings)
            else:
                firings, batch_I, batch_pretrace = self.generateFirings()
                self.batchReset(firings, batch_I, batch_pretrace)

            if self.memorySavingMode:
                for t in range(self.batchsize_dt):
                    self.updateLowMemory(t=t, offset=batch*self.batchsize_dt, learning=learning)
            else:
                for t in range(self.batchsize_dt):
                    self.update(t=t, offset=batch*self.batchsize_dt, learning=learning)

            v_spikes[:, batch*self.batchsize_dt:(batch + 1)*self.batchsize_dt] = self.v_spikes[:, :-1]

        return v_spikes

    def updateLowMemory(self, t, parallel_id=0, It=False, offset=0, learning=True):

        """
            Performs one time step in the simulation, used for low memory simulations
        """

        idx = self.mso_idx[parallel_id]

        if not It:
            It = np.sum(self.w[idx, :]*self.I[idx, :], axis=1)

        c = self.__class__    # Shortcut to class variables

        if self.memorySavingMode == 1:
            presynaptic_spikes = self.delayed_firings[idx, :, t + 1]
        elif self.memorySavingMode == 2:
            presynaptic_spikes = np.zeros((len(idx), self.input_neurons))
            presynaptic_spikes[self.delayed_firings[parallel_id][0][t + 1], self.delayed_firings[parallel_id][1][t + 1]] = 1.0

        # Update LIF neuron
        refrac = ((t + offset - self.last_spike[idx]) < (c.REFRACTORY_PERIOD/self.dt))
        self.v[idx] = refrac*self.v[idx] + (1 - refrac)*self.euler(self.dv, self.v[idx], [It], self.dt)

        postsynaptic_spikes = np.zeros(self.MSO_neurons).astype(bool)
        postsynaptic_spikes[idx] = self.v[idx] > self.vt
        self.v[postsynaptic_spikes] = 0.0
        self.last_spike[postsynaptic_spikes] = t + offset
        self.v_spikes[idx, t] = postsynaptic_spikes[idx]

        self.I[idx, :] = self.euler(self.dI, self.I[idx], [], self.dt) + presynaptic_spikes*(1 / c.TAU_S)

        if learning:

            # Output trace
            self.pretrace[idx, :] = self.euler(self.dltp, self.pretrace[idx, :], [], self.dt) + presynaptic_spikes*c.A_LTP
            self.posttrace[idx] = self.euler(self.dltd, self.posttrace[idx], [], self.dt) + postsynaptic_spikes[idx]*c.A_LTD  # *self.u

            # Update synapses
            self.w[idx] += (self.pretrace[idx, :]*postsynaptic_spikes[idx, np.newaxis] - (presynaptic_spikes*self.posttrace[idx, np.newaxis]))*(self.w[idx] > 0)
            self.w[idx] = np.clip(self.w[idx], 0, self.__class__.SATURATION)

    def update(self, t, parallel_id=0, It=False, offset=0, learning=True):
        """
            Performs one time step in the simulation, high memory usage (assumes
            pre-computed I and pretrace values )
        """

        idx = self.mso_idx[parallel_id]

        if not It:
            It = np.sum(self.w[idx, :]*self.I[idx, :, t], axis=1)

        c = self.__class__    # Shortcut to class variables

        # Update LIF neuron
        refrac = ((t + offset - self.last_spike) < (c.REFRACTORY_PERIOD/self.dt))
        self.v[idx] = refrac*self.v[idx] + (1 - refrac)*self.euler(self.dv, self.v[idx], [It], self.dt)

        postsynaptic_spikes = np.zeros(self.MSO_neurons).astype(bool)
        postsynaptic_spikes = self.v[idx] > self.vt
        self.v[postsynaptic_spikes] = 0.0
        self.last_spike[postsynaptic_spikes] = t + offset
        self.v_spikes[idx, t] = postsynaptic_spikes[idx]

        if learning:
            spikes = self.delayed_firings[idx, :, t + 1]

            # Output trace
            self.posttrace[idx] = self.euler(self.dltd, self.posttrace[idx], [], self.dt) + postsynaptic_spikes[idx]*c.A_LTD  # *self.u

            # Update synapses
            self.w[idx] += (self.pretrace[idx, :, t]*postsynaptic_spikes[idx, np.newaxis] - (spikes*self.posttrace[idx, np.newaxis]))*(self.w[idx] > 0)
            self.w[idx] = np.clip(self.w[idx], 0, self.__class__.SATURATION)

    def generateFirings(self):
        """
            Generates firings for a single batch. Returns an n x batchsize array whose entries
            indicate whether or not neuron n fired at time t. Incorporates refractory period.
        """

        batchsize_dt, dt, input_neurons, refractory_period = [self.batchsize_dt, self.dt, self.input_neurons, self.__class__.REFRACTORY_PERIOD]

        if not self.memorySavingMode:
            batch_I = np.zeros((input_neurons, batchsize_dt + self.I_mask_overlay))
            batch_pretrace = np.zeros((input_neurons, batchsize_dt + self.pretrace_mask_overlay))
        batch_firings = np.zeros((input_neurons, batchsize_dt))

        i = int(0.5*input_neurons)
        batch_firings[:i, :] = self.p_values_left*dt > np.random.rand(i, batchsize_dt)
        batch_firings[i:, :] = self.p_values_right*dt > np.random.rand(i, batchsize_dt)

        (js, t_firings) = np.where(batch_firings > 0)

        # Add refractory period to generated spikes
        i = 0
        while(i < len(t_firings)):

            if (i == 0 or js[i] != js[i - 1]) and np.sum(self.prevFirings[js[i]]) > 0:
                batch_firings[js[i], t_firings[i]] = 0.0
                i += 1
                continue

            # This firing was deactivated in a previous run
            if batch_firings[js[i], t_firings[i]] == 0.0:
                i += 1
                continue

            if not self.memorySavingMode:
                batch_I[js[i], (t_firings[i]):(t_firings[i] + self.I_mask_overlay)] += self.I_mask
                batch_pretrace[js[i], (t_firings[i]):(t_firings[i] + self.pretrace_mask_overlay)] += self.pretrace_mask

            # Remove potential firings within refractory period
            n = i + 1
            while n < len(t_firings) and js[n] == js[i] and t_firings[n] - t_firings[i] < (refractory_period / dt):
                batch_firings[js[n], t_firings[n]] = 0.0
                n += 1

            i += 1

        if self.memorySavingMode:
            return batch_firings
        else:
            return batch_firings, batch_I, batch_pretrace

    def euler(self, f, x, args, dt):
        """
            Computes the euler integration of a given f with input valus x and args args
        """
        return x + dt*f(x, *args)


def main():

    """ Can be used to run a simulation from the shell """

    storeData = False
    memorySavingModeFlag = 0
    parallel = 1
    batchmemprofile = False
    msomemprofile = False
    speedtest = False
    speedtestmso = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:v",
                                   ["timesteps=", "batchsize=", "vt=", "input_neurons=", "MSO_neurons=", "dt=", "storeData",
                                    "memorySavingMode=", "ITDoptions=", "speedTestMSO", "parallel=", "T=", "profileBatch", "profileMSOneurons", "speedTest"])
    except getopt.GetoptError as e:
        print str(e)
        sys.exit(2)

    for opt, arg in opts:
        print "\t{}\t{}".format(opt, arg)
        if opt in ("--timesteps"):
            timesteps = int(arg)
        elif opt in ("--batchsize"):
            batchsize = int(arg)
        elif opt in ("--vt"):
            vt = float(arg)
        elif opt in ("--input_neurons"):
            input_neurons = int(arg)
        elif opt in ("--MSO_neurons"):
            MSO_neurons = int(arg)
        elif opt in ("--dt"):
            dt = float(arg)
        elif opt in ("--storeData"):
            storeData = True
        elif opt in ("--ITDoptions"):
            ITDoptions = eval(arg)
        elif opt in ("--T"):
            T = float(arg)
        elif opt in ("--memorySavingMode"):
            memorySavingModeFlag = int(arg)
        elif opt in ("--parallel"):
            parallel = int(arg)
        elif opt in ("--profileBatch"):
            if mem_profiler_not_available:
                raise ValueError('memory_profiler not installed')
            batchmemprofile = True
        elif opt in ("--profileMSOneurons"):
            if mem_profiler_not_available:
                raise ValueError('memory_profiler not installed')
            msomemprofile = True
        elif opt in ("--speedTest"):
            speedtest = True
        elif opt in ("--speedTestMSO"):
            speedtestmso = True

    msoneurons = [1, 2, 4, 8, 20, 50, 100]
    batchsizes = [80, 100, 200, 500, 1000, 2000]

    if batchmemprofile:
        mem_batch = np.zeros(len(batchsizes))
        for i, batchsize in enumerate(batchsizes):
            sim = MSOModel(
                timesteps=2*batchsize,
                batchsize=batchsize,
                vt=vt,
                input_neurons=input_neurons,
                MSO_neurons=MSO_neurons,
                dt=dt,
                storeData=False,
                ITDoptions=ITDoptions,
                T=T,
                memorySavingMode=memorySavingModeFlag,
                parallel=parallel
            )

            sim.testMode = True

            mem_batch[i] = np.max(memory_usage(sim.run))
            print mem_batch[i]
        print batchsizes
        print mem_batch
    elif msomemprofile:
        mem_mso = np.zeros(len(msoneurons))
        for i, msoinput in enumerate(msoneurons):
            sim = MSOModel(
                timesteps=timesteps,
                batchsize=batchsize,
                vt=vt,
                input_neurons=input_neurons,
                MSO_neurons=msoinput,
                dt=dt,
                storeData=False,
                ITDoptions=ITDoptions,
                T=T,
                memorySavingMode=memorySavingModeFlag,
                parallel=parallel
            )

            sim.testMode = True

            mem_mso[i] = np.max(memory_usage(sim.run))
            print mem_mso
            print mem_mso[i]
        print mem_mso

    elif speedtest:
        speed_batch = np.zeros(len(batchsizes))

        for i, batchsize in enumerate(batchsizes):
            sim = MSOModel(
                timesteps=2*max(batchsizes),
                batchsize=batchsize,
                vt=vt,
                input_neurons=input_neurons,
                MSO_neurons=MSO_neurons,
                dt=dt,
                storeData=False,
                ITDoptions=ITDoptions,
                T=T,
                memorySavingMode=memorySavingModeFlag,
                parallel=parallel
            )

            sim.testMode = True

            start = time.time()
            speed_batch[i] = sim.run()
            end = time.time()

            speed_batch[i] = end - start
            print speed_batch[i]

        print speed_batch

    elif speedtestmso:
        speed_mso = np.zeros(len(msoneurons))

        for i, nr_mso in enumerate(msoneurons):
            sim = MSOModel(
                timesteps=timesteps,
                batchsize=batchsize,
                vt=vt,
                input_neurons=input_neurons,
                MSO_neurons=nr_mso,
                dt=dt,
                storeData=False,
                ITDoptions=ITDoptions,
                T=T,
                memorySavingMode=memorySavingModeFlag,
                parallel=parallel
            )

            sim.testMode = True

            start = time.time()
            speed_mso[i] = sim.run()
            end = time.time()

            speed_mso[i] = end - start
            print speed_mso[i]

        print speed_mso

    else:
        sim = MSOModel(
            timesteps=timesteps,
            batchsize=batchsize,
            vt=vt,
            input_neurons=input_neurons,
            MSO_neurons=MSO_neurons,
            dt=dt,
            storeData=storeData,
            ITDoptions=ITDoptions,
            T=T,
            memorySavingMode=memorySavingModeFlag,
            parallel=parallel
        )

        sim.run()


   # sim.run()

if __name__ == "__main__":

    main()
