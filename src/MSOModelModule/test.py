""" The test module contains helper functions that can be used to test the MSOModel and analyse its results. """

import numpy as np
from core import MSOModel
from os import listdir
import matplotlib.gridspec as gridspec
import json
import gzip
import matplotlib as mpl
import datetime
import os
import sys
import getopt


def ITDcorrelation(path, t, ITDoptions):
    """
        ITD correlation runs an MSO model stored in a particular path for the given ITD options
        to test its responses for 10 ms. It stores the spikes of the test in a test folder that is
        created in the folder where the simulation is stored.

        :param t: The time at which the simulation should be tested
        :param ITDoptions: an array that contains all ITDs that should be tested
        :param path: a string that refers to the path where the simulation is stored


        .. note::
            The parameter t is in timesteps, not in milliseconds. You compute the time you want to test
            the simulation on as t \ dt, where dt is the stepsize. This requires the existence of a file
            (t \ dt - 1)_w.json.gz in the data_per_batch section of the simulation folder
   
    """
    print t

    sim = MSOModel.loadLearnedModel(path, t - 1)

    timestamp = str(datetime.datetime.now().time())[:-7].replace(':', '-')  # Give similar simulations a unique folder

    npath = "{}/tests/{}".format(path, timestamp)
    if not os.path.exists(npath):
        os.makedirs(npath)

    params = {'t': t, 'ITDoptions': ITDoptions}

    with open('{}/parameters.txt'.format(npath), 'w') as outfile:
            json.dump(params, outfile)

    del sim

    for ITD in ITDoptions:
        sim = MSOModel.loadLearnedModel(path, t - 1)
        sim.setITD(ITD)
        v_spikes = sim.test(timesteps=10000)

        with gzip.open('{}/vspikes_{}.json.gz'.format(npath, ITD).replace('.', '-'), 'w') as outfile:
            json.dump(v_spikes.tolist(), outfile)
        del sim


def plotITDTest(path, t, ITDs):

    from matplotlib import pyplot as plt

    sim = MSOModel.loadLearnedModel(path, t - 1)

    left = sim.p_values_left
    right = sim.p_values_right

    time = 10.0
    t_range = sim.t_range[:sim.batchsize/sim.dt]

    #ITDs = [-0.5, 0.0, 0.5]
    #ITDs = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    ITD_results = [[] for _ in range(len(ITDs))]

    timestamp = str(datetime.datetime.now().time())[:-7].replace(':', '-')  # Give similar simulations a unique folder
    npath = "{}/tests/ITDtest/{}".format(path, timestamp)
    if not os.path.exists(npath):
        os.makedirs(npath)

    #for i, ITD in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
    for i, ITD in enumerate(ITDs):
        print "\tTesting {}".format(ITD)
        sim = MSOModel.loadLearnedModel(path, t - 1)
        sim.setITD(ITD)
        v_spikes = sim.test(10000)
        ITD_results[i] = [v_spikes.copy(), sim.p_values_left.copy(), sim.p_values_right.copy()]

        with gzip.open('{}/ITD_{}.json.gz'.format(npath, str(ITD).replace('.', '-')), 'w') as outfile:
            json.dump(v_spikes.tolist(), outfile)

    T = sim.T

    for neuron in range(sim.MSO_neurons):

        plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3 + len(ITDs), 2)
        ax = [0]*(gs.get_geometry()[0]*gs.get_geometry()[1])

        ax[0] = plt.subplot(gs[0, :])
        ax[0].plot(t_range[:time/sim.dt], left[:time/sim.dt])
        ax[0].plot(t_range[:time/sim.dt], right[:time/sim.dt])
        ax[0].set_title('ITD: {} ms'.format(sim.ITD))

        ax[1] = plt.subplot(gs[1, 0])
        left_idx = 0.5*sim.input_neurons
        ax[1].set_title('Delays after learning (left)')
        ax[1].hist(sim.delays[neuron, :left_idx], 400, range=(1.5, 7.5), weights=sim.w[neuron, :left_idx])
        ax[1].set_xlabel('Delay (ms)')
        ax[1].set_ylabel('Delays / bin')

        ax[2] = plt.subplot(gs[1, 1])
        ax[2].set_title('Delays after learning (right)')
        ax[2].hist(sim.delays[neuron, left_idx:], 400, range=(1.5, 7.5), weights=sim.w[neuron, left_idx:])
        ax[2].set_xlabel('Delay (ms)')
        ax[2].set_ylabel('Delays / bin')

        for i, ITD in enumerate(ITDs):
            if i > 0:
                ax[3 + i] = plt.subplot(gs[2 + i, 0], sharey=ax[3])
            else:
                ax[3 + i] = plt.subplot(gs[2 + i, 0])

            v_spikes, left_n, right_n = ITD_results[i]
            spikes = np.where(v_spikes[neuron] == 1.0)[0]
            ax[3 + i].hist(np.array(spikes)*sim.dt % T, T/sim.dt, range=[0, T])
            ax[3 + i].set_title('Total spikes: {}'.format(len(spikes)))

            ax[4 + i] = plt.subplot(gs[2 + i, 1])
            ax[4 + i].set_title('ITD: {} ms'.format(ITD))
            ax[4 + i].plot(t_range[:time/sim.dt], left_n[:time/sim.dt])
            ax[4 + i].plot(t_range[:time/sim.dt], right_n[:time/sim.dt])

        plt.tight_layout()


def plotDelays(path, maxVal):

    from matplotlib import pyplot as plt

    with open('{}/delays.txt'.format(path), 'r') as f:
        delays = np.array(json.loads(f.readlines()[0]))

    with open('{}/simulation_settings.txt'.format(path), 'r') as f:
        params = dict(json.loads(f.readlines()[0]))

    with open('{}/parameters.txt'.format(path), 'r') as f:
        c = dict(json.loads(f.readlines()[0]))

    #with gzip.open('{}/data_per_batch/{}_vspikes.json.gz'.format(path, maxVal - 1), 'r') as f:
    #    vspikes = np.array(json.loads(f.readlines()[0]))

    stepsize = params['batchsize'] / params['dt']
    t_range = np.arange(0, maxVal + stepsize, stepsize) - 1

    MSO_neurons, nr_neurons = delays.shape

    binrange = (c['DELAY_MU'] - 3*c['DELAY_SIGMA'], c['DELAY_MU'] + 3*c['DELAY_SIGMA'])

    J = np.zeros((MSO_neurons, nr_neurons, len(t_range)))
    vspikes = [[] for _ in range(MSO_neurons)]

    for i, t in enumerate(t_range):
        if i == 0:
            J[:, :, i] = 1.0
            continue

        with gzip.open('{}/data_per_batch/{}_w.json.gz'.format(path, int(t)), 'r') as f:
            J[:, :, i] = np.array(json.loads(f.readlines()[0]))


    vspikes = np.zeros((MSO_neurons, 1))

    for i in range(min(MSO_neurons, 5))[:1]:
        print i
        plt.figure()

        colors = np.linspace(0.0, 1.0, params['input_neurons'])
        mymap = plt.get_cmap("spring")
        # get the colors from the color map
        my_colors = mymap(colors)
        mymap = plt.get_cmap("winter")
        my_colors_warm = mymap(colors)

        bins = np.linspace(min(delays[i]), max(delays[i]), params['input_neurons'])
        delays_id = [np.where(delay >= bins)[0][-1] for delay in delays[i]]

        gs = gridspec.GridSpec(10, 3)
        ax = [plt.subplot(gs[:7, :]), plt.subplot(gs[8, :]), plt.subplot(gs[9, :])]

        for j in range(nr_neurons):
            if j < nr_neurons / 2:
                ax[0].plot(t_range*params['dt']/1000.0, J[i, j, :], alpha=0.8, color=my_colors[delays_id[j]])
            else:
                ax[0].plot(t_range*params['dt']/1000.0, J[i, j, :], alpha=0.8, color=my_colors_warm[delays_id[j]])

        ax[0].set_ylabel(r'Synaptic strength $w$')
        ax[0].set_xlabel('Time (s)')

        ax[0].set_xlim([0, max(t_range)*params['dt']/1000.0])

        cmap = mpl.cm.spring
        norm = mpl.colors.Normalize(vmin=min(delays[i]), vmax=max(delays[i]))

        cb1 = mpl.colorbar.ColorbarBase(ax[1], cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')

        cmap = mpl.cm.winter
        norm = mpl.colors.Normalize(vmin=min(delays[i]), vmax=max(delays[i]))

        cb2 = mpl.colorbar.ColorbarBase(ax[2], cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
        ax[2].set_xlabel('Delays (ms)')

        plt.tight_layout()

        plt.figure()

        gs = gridspec.GridSpec(4, 2)
        ax = [0]*6
        ax[0] = plt.subplot(gs[:2, 0])
        ax[1] = plt.subplot(gs[:2, 1], sharey=ax[0])
        #ax[2] = plt.subplot(gs[1, :])
        ax[3] = plt.subplot(gs[2:, 0], sharey=ax[0])
        ax[4] = plt.subplot(gs[2:, 1], sharey=ax[0])
        #ax[5] = plt.subplot(gs[3, :])

        left_idx = int(delays.shape[1]/2.0)

        ax[0].set_title('Delays before learning (left)')
        ax[0].hist(delays[i, :left_idx], 400, range=binrange, weights=J[i, :left_idx, 0])
        ax[0].set_xlabel('Delay (ms)')
        ax[0].set_ylabel('Delays / bin * weight')

        ax[1].set_title('Delays before learning (right)')
        ax[1].hist(delays[i, left_idx:], 400, range=binrange, weights=J[i, left_idx:, 0])
        ax[1].set_xlabel('Delay (ms)')
        ax[1].set_ylabel('Delays / bin * weight')

        #v_spikes = np.array(vspikes[i])
        #firstvspikes = v_spikes[v_spikes < stepsize]
        #ax[2].hist(np.array(firstvspikes)*params['dt'] % params['T'], params['T']/params['dt'], range=[0, params['T']])

        ax[3].set_title('Delays after learning (left)')
        ax[3].hist(delays[i, :left_idx], 400, range=binrange, weights=J[i, :left_idx, len(t_range) - 1])
        ax[3].set_xlabel('Delay (ms)')
        ax[3].set_ylabel('Delays / bin * weight')

        ax[4].set_title('Delays after learning (right)')
        ax[4].hist(delays[i, left_idx:], 400, range=binrange, weights=J[i, left_idx:, len(t_range) - 1])
        ax[4].set_xlabel('Delay (ms)')
        ax[4].set_ylabel('Delays / bin * weight')

        for axis in ax:
            if axis != 0:
                axis.locator_params(axis='y', nbins=4)

        plt.tight_layout()

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["path=", "timesteps=", "plotITDtest=", "plotDelays", "ITDcorrelation="])
    except getopt.GetoptError as e:
        print str(e)
        sys.exit(2)

    ITDcorrFlag = False
    plotITDFlag = False
    plotDelaysFlag = False
    t = False

    for opt, arg in opts:
        if opt in ("--timesteps"):
            t = int(arg)
        elif opt in ("--path"):
            path = arg
        elif opt in ("--ITDcorrelation"):
            ITDoptions = eval(arg)
            ITDcorrFlag = True
        elif opt in ("--plotITDtest"):
            ITDoptionsPlot = eval(arg)
            plotITDFlag = True
        elif opt in ("--plotDelays"):
            plotDelaysFlag = True

    if not t:
        t = np.sort([int(f.split('_')[0]) for f in listdir('{}/data_per_batch/'.format(path)) if '_w.json.gz' in f])[-1] + 1

    if ITDcorrFlag:
        ITDcorrelation(path, t, ITDoptions)
    if plotITDFlag:
        plotITDTest(path, t, ITDoptionsPlot)
    if plotDelaysFlag:
        plotDelays(path, t)
    if plotITDFlag or plotDelaysFlag:
        from matplotlib import pyplot as plt
        plt.show()
