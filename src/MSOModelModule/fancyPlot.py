"""The fancyPlot module contains a set of useful functions for plotting the results"""

import sys
sys.path.append("..")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from MSOModelModule.core import MSOModel
import gzip
import json
from os import listdir
from os.path import isdir
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

rcParams['axes.labelsize'] = 24  # 'large'
rcParams['axes.titlesize'] = 24  # 'large'
rcParams['xtick.labelsize'] = 24  # 'large'  # 'large'
rcParams['ytick.labelsize'] = 24  # 'large'
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'


def basiccolors():
    """ Returns array containing set of colors used for plots"""
    return ['#0083d9', '#008b4e', '#ffac00', '#DD1E2F', '#192823']


def color_variant(hex_color, brightness_offset=1):
    """ Changes hexadecimal color value to a lighter or darker version"""
    rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int]  # make sure new values are between 0 and 255
    l = [hex(i)[2:] if len(hex(i)[2:]) > 1 else '0' + hex(i)[2:] for i in new_rgb_int]
    return "#" + "".join(l)


def plotBar(x, y, ax, c, label="", width=0.2):
    """ Plots a bar plot with an error bar
        :param x: x-value of bars
        :param y: a 2d array, first dimension is same as x
        :param ax: reference to the axes at which the bar should be plotted
        :param label: string with the label given to each bar
        :param width: width of the barplot
    """
    mean = np.array([np.mean(i) for i in y])
    error = np.array([np.std(i) for i in y])

    if np.sum(error) > 0:
        # If there is a variance, plot the edge in a darker color of the original barplot
        dark_color = color_variant(c, -100)
        ax.bar(x - width/2.0, mean, width, yerr=error, color=c,  edgecolor='none',
               error_kw=dict(elinewidth=2, capthick=2.0, ecolor=dark_color), label=label)
    else:
        ax.bar(x - width/2.0, mean, width, color=c, edgecolor='none', label=label)


def makeColorMap(colors):
    """
        Creates a colormap for a given array with colors (can be used for e.g. imshow() plots)
    """
    c = mcolors.ColorConverter().to_rgb

    rgb_list = np.zeros((len(colors), 3))
    for i in range(len(colors)):
        rgb_list[i, :] = c(colors[i])

    rgb_list += 0.1

    bins = np.linspace(0, 1, len(colors))

    cdict_array = np.zeros((3, len(bins), 3))

    for i in range(3):
        cdict_array[i, :, 0] = bins
        cdict_array[i, :, 1] = rgb_list[:, i]
        cdict_array[i, :, 2] = rgb_list[:, i]

    cdict = {}
    cdict['red'] = tuple([tuple(x) for x in cdict_array[0]])
    cdict['green'] = tuple([tuple(x) for x in cdict_array[1]])
    cdict['blue'] = tuple([tuple(x) for x in cdict_array[2]])

    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def loadWeights(path, t):
    """
        Loads the weight matrix of timestep t located in the directory path 
    """
    with gzip.open('{}/data_per_batch/{}_w.json.gz'.format(path, int(t - 1)), 'r') as f:
        w = np.array(json.loads(f.readlines()[0]))
    return w


def plotMeanAndErr(x, y, ax, c, label="", alpha=0.3):
    """
        Plots the mean and variance of y 
        :param x: x-value of plot 
        :param y: a 2d array, first dimension is same as x
        :param ax: reference to the axes at which the plot should be plotted
        :param label: string with the label given to the plot
        :param alpha: alpha of the plot
    """
    mean = np.array([np.mean(i) for i in y])
    error = np.array([np.std(i) for i in y])
    ax.fill_between(x, mean-error, mean + error, color=c, alpha=alpha, zorder=1)
    ax.plot(x, mean, color=c, lw=2.0, label=label, zorder=10)
    #ax.plot(x, mean - error, color=c, label="_", ls='-', lw=1.5, alpha=0.5, zorder=5)
    #ax.plot(x, mean + error, color=c, label="_", ls='-', lw=1.5, alpha=0.5, zorder=5)


def computeMonaural(w):
    """
        Computes the percentage of monaural selective MSO neurons for a given weight matrix
    """
    return np.sum((np.sum(w[:, 250:] > 0, axis=1) == 0)*1.0 + (np.sum(w[:, :250] > 0, axis=1) == 0)*1.0 >= 1.0)/float(w.shape[0])*100


def vectorStrength(spikes, T, dt, sparse=False):
    """
        Computes the vector strength for period T for a particular spike sequence

        :param sparse: If False, spikes is assumed to be sparse array with ones at \
        times a spike occurred. If True, spikes is assumed to be a list of indices where
        the MSO neuron has spiked.

        Only works for spike trains of single MSO neurons! 
    """
    if not sparse:
        ts = np.where(spikes == 1.0)[0]
    else:
        ts = spikes
    phases = [(t*dt % T)/T*2*np.pi for t in ts]
    if len(phases) == 0:
        return 0
    vectorStrength = (1.0 / len(ts))*np.sqrt(np.sum([np.cos(phase) for phase in phases])**2 + np.sum([np.sin(phase) for phase in phases])**2)
    return vectorStrength


def cleanAxes(ax):
    """
        Removes ugly spines and ticks of a plot
    """
    for i in range(len(ax)):
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].get_xaxis().tick_bottom()
        ax[i].get_yaxis().tick_left()
        ax[i].tick_params('both', length=5, width=2, which='major')
        ax[i].tick_params('both', length=5, width=1, which='minor')


def maxTicks(ax, (max_xticks, max_yticks)):
    """
        Sets the maximum number of ticks on the x and y axis of ax x 
    """
    for i in range(len(ax)):
        yloc = plt.MaxNLocator(max_yticks)
        xloc = plt.MaxNLocator(max_xticks)
        ax[i].yaxis.set_major_locator(yloc)
        ax[i].xaxis.set_major_locator(xloc)


def plotWeightEvolvement(paths, labels, timesteps, colors):
    """
        Plots the weight evolvement of a simulation stored in path over a range of timesteps 

        :param paths: lists of strings that point to the directory where the simulation is stored
        :param labels: the labels that indicate which path has which label
        :param timesteps: array with all the timesteps should (approximately) be plotted
        :param colors: colors in which each path should be plotted
    """
    fig, ax = plt.subplots(2, figsize=(10, 8))

    x = (timesteps*0.005/1000.0).astype(int)
    sums = [[0]*len(timesteps) for m in range(len(paths))]
    delays = [[1]*len(timesteps) for m in range(len(paths))]

    batchsizes = np.zeros(len(paths))
    for i, path in enumerate(paths):
        with open('{}/simulation_settings.txt'.format(path), 'r') as f:
            true = True
            sim = eval(f.readlines()[0])
        batchsizes[i] = sim['batchsize'] / sim['dt']

    for i, t in enumerate(timesteps):
        for j, path in enumerate(paths):
            t = int(t/int(batchsizes[j]))*batchsizes[j]
            try:
                w = loadWeights(path, int(t))
                sums[j][i] = np.sum(w, axis=1)
                delays[j][i] = np.sum(w > 0, axis=1)
            except Exception as e:
                print e
                pass

    for p in range(len(paths)):
        plotMeanAndErr(x, sums[p], ax[0], colors[p], label=labels[p], alpha=0.2)
        plotMeanAndErr(x, np.array(sums[p])/np.array(delays[p]), ax[1], colors[p], alpha=0.3)

    # ax[0].plot([0, timesteps[-1]*0.005/1000.0], [500, 500], lw=3.0, color=colors[-1], label=labels[-1])
    ax[0].legend(loc=1, frameon=False, ncol=2, fontsize=23)

    ax[0].set_xlim([0, timesteps[-1]*0.005/1000.0])
    ax[0].set_xlabel('Time (s)')
    ax[1].set_xlabel('Time (s)')
    ax[0].set_ylabel('Total weight $w$')
    ax[1].set_ylabel('Avg $w$ / neuron')
    ax[1].plot(x, [3.0]*len(x), ls='-', color='gray', lw=2.0, alpha=0.5, zorder=1)
    cleanAxes(ax)
    plt.tight_layout()


def plotWeightEvolvementsClasses(paths, colors, labels, ts, ids, nr_sim):
    """
        Plots weight evolvement over paths that may be part of similar classes (e.g. if you have
        two paths that both use the same input these are combined)
    """
    fig, ax = plt.subplots(3, figsize=(11, 11))

    x = (ts*0.005/1000.0).astype(int)
    total = len(nr_sim)
    sums = [[0]*len(ts) for m in range(total)]
    delays = [[1]*len(ts) for m in range(total)]
    monaural = [[[] for a in range(len(ts))] for m in range(total)]

    for i, t in enumerate(ts):
        t = int(t/int(1000/0.005))*(1000/0.005)
        for j, path in enumerate(paths):
            try:
                w = loadWeights(path, int(t))
                sums[ids[j]][i] += np.sum(w, axis=1)
                delays[ids[j]][i] += np.sum(w > 0, axis=1)
                monaural[ids[j]][i].append(computeMonaural(w))
            except Exception as e:
                pass

    for p in range(total):
        plotMeanAndErr(x, np.array(sums[p])/nr_sim[p], ax[0], colors[p % len(colors)], label=labels[p])
        plotMeanAndErr(x, np.array(sums[p])/np.array(delays[p]), ax[1], colors[p % len(colors)])
        plotMeanAndErr(x, monaural[p], ax[2], colors[p % len(colors)])

    ax[0].legend(loc=1, frameon=False, ncol=2, fontsize=15)

    ax[0].set_xlim([0, ts[-1]*0.005/1000.0])
    ax[0].set_xlabel('Time (ms)')
    ax[1].set_xlabel('Time (ms)')
    ax[2].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Total weights $w$')
    ax[2].set_ylabel('Monaural neurons (%)')
    ax[1].set_ylabel('Avg $w$ / neuron')
    ax[1].plot(x, [3.0]*len(x), ls='-', color='gray', lw=2.0, alpha=0.5, zorder=1)
    ax[1].set_ylim([0, 3.5])
    ax[2].set_ylim([0, 100])

    cleanAxes(ax)
    maxTicks(ax, (4, 4))

    plt.tight_layout()


def plotMonauralBarplot(paths, colors, labels, t, ids, nr_sim):
    """
        Plots the number of monaural MSO neurons at time t in the given simulations
        Combines paths in classes for given class ids and nr_sim
    """
    fig, ax = plt.subplots(1, figsize=(5, 3))
    total = len(nr_sim)
    monaural = [[] for a in range(len(nr_sim))]

    t = int(t/int(1000/0.005))*(1000/0.005)
    for j, path in enumerate(paths):
        try:
            w = loadWeights(path, int(t))
            monaural[ids[j]].append(computeMonaural(w))
        except Exception as e:
            pass

    for p in range(len(nr_sim) - 1):
        plotBar(np.array([p]), np.array([monaural[p]]), ax, colors[p], label=labels, width=0.5)

    ax.set_ylim([0, 100])
    ax.set_ylabel('Monaural neurons (%)')
    cleanAxes([ax])
    maxTicks([ax], (4, 4))
    ax.set_xticks(range(len(labels[:-1])))
    ax.set_xticklabels(labels[:-1])

    plt.tight_layout()


def plotDistributionTaskRandom(labels, colors, nr_per_itd, itdoptions, width=0.2, zero=True):
    """
        Plots a barplot of a histogram list (nr_per_itd) on x-axis value itdoptions
    """

    if zero:
        r = len(labels)
    else:
        r = len(labels) - 1

    fig, ax = plt.subplots(1, r, figsize=(12, 3.0))

    for i in range(r):
        plotBar(np.array(itdoptions[i]), np.array(nr_per_itd[i]) / np.sum(nr_per_itd[i])*100, ax[i], colors[i % len(colors)], label=labels[i], width=width)
        ax[i].set_xlim([min(itdoptions[i]), max(itdoptions[i])])
        ax[i].set_title(labels[i])
        ax[i].set_ylim([0, 100])
        ax[i].set_xlabel('Best ITD (ms)')
        ax[i].set_title(labels[i], y=0.85, x=0.5)

    cleanAxes(ax)
    if min(itdoptions[i]) > -1:
        maxTicks(ax, (3, 5))
    else:
        maxTicks(ax, (4, 5))
    ax[0].set_ylabel('% MSO neurons')

    plt.tight_layout()


def loadMeasurements(path, test_nr, precomputed=False):
    """
        Loads the mean firing rate and vector strength of a particular simulation.

        :param test_nr: indicates which test should be loaded (in alphabetical order)
        :param precomputed: whether the MFR and VS are already precomputed or should be loaded
        from the recorded spike trains (much slower)

    """
    with open('{}/simulation_settings.txt'.format(path), 'r') as f:
        true = True
        sim = eval(f.readlines()[0])
        dt = sim['dt']
        T = sim['T']

    path = '{}/tests'.format(path)
    directory = [f for f in listdir(path) if isdir('{}/{}'.format(path, f))][test_nr]

    with open('{}/{}/parameters.txt'.format(path, directory), 'r') as f:
        params = json.loads(f.readlines()[0])
        itdoptions = params['ITDoptions']
        t = params['t']

    mfr = [0]*len(itdoptions)
    vs = [0]*len(itdoptions)

    for j, itd in enumerate(itdoptions):
        if precomputed:
            npath = path + '/{}/mfr_{}.json.gz'.format(directory, itd)
            with gzip.open(npath, 'r') as f:
                mfr[j] = np.array(json.loads(f.readlines()[0]))

            npath = path + '/{}/vs_{}.json.gz'.format(directory, itd)
            with gzip.open(npath, 'r') as f:
                vs[j] = np.array(json.loads(f.readlines()[0]))
        else:
            npath = path + '/{}/vspikes_{}.json.gz'.format(directory, itd).replace('.', '-')
            with gzip.open(npath, 'r') as f:
                v_spikes = np.array(json.loads(f.readlines()[0]))

            nrneur = v_spikes.shape[0]
            mfr[j] = np.sum(v_spikes, axis=1)/10.0
            vs[j] = [0]*nrneur

            for neuron in range(v_spikes.shape[0]):
                vs[j][neuron] = vectorStrength(v_spikes[neuron], T, dt)

            vs[j] = np.array(vs[j])

        # Set insignificant vectors to 0
        rayleigh = np.exp(-(mfr[j]*10)*vs[j]**2)
        vs[j][np.where(rayleigh > 0.01)[0]] = 0.0
        vs[j][np.where(mfr[j] <= 5.0)[0]] = 0.0

    if abs(itdoptions[0]) > abs(itdoptions[-1]):
        mfr.append(mfr[0])
        vs.append(vs[0])
        itdoptions = np.append(itdoptions, -itdoptions[0])
    elif abs(itdoptions[0]) < abs(itdoptions[-1]):
        mfr.insert(0, mfr[-1])
        vs.insert(0, vs[-1])
        itdoptions = np.insert(itdoptions, 0, -itdoptions[-1])

    return itdoptions, mfr, vs, t


def loadSpikeTrain(path, test_nr):
    """
        Loads the spikes at ITD 0.0 for a path in test test_nr (tests are loaded in \
            alphabetical order)
    """

    with open('{}/simulation_settings.txt'.format(path), 'r') as f:
        true = True
        sim = eval(f.readlines()[0])
        dt = sim['dt']
        T = sim['T']

    path = '{}/tests'.format(path)
    directory = [fi for fi in listdir(path) if isdir('{}/{}'.format(path, fi))][test_nr]

    with gzip.open('{}/{}/vspikes_0-0-json-gz'.format(path, directory), 'r') as f:
        v_spikes = np.array(json.loads(f.readlines()[0]))

    return v_spikes, T, dt

def plotModulationWithoutBefore(mod_per_itd, labels, ptype, colors):

    """
        Plots a barplot of a range of values (mot per ITD) with given labels

        :param ptype: Stringt of what should be plotted on the y-axis
    """

    fig, ax = plt.subplots(1, figsize=(5, 3))
    modflattened = [[item for sublist in l for item in sublist] for l in mod_per_itd]
    ncat = len(modflattened)
    for i in range(ncat):
        plotBar(np.array([i]), [np.array(modflattened[i])], ax, colors[i % len(colors)], width=0.5)

    ax.set_ylim([0, 1])
    ax.set_xticks([])

    cleanAxes([ax])
    ax.set_ylabel(ptype)

    xticks = np.array(range(len(labels)))
    xticks[-1] += 0.2
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)

def plotITDCurves(paths, colors, precomputed=False, test_nr=False, split=False, refvals=False, includePhaseLock=False):
    """
        Plots the MFR and VS curves for all simulations stored in paths

        :param paths: list of strings where simulations are stored
        :param colors: list of the colors in which of above mentioned paths should be plotted
        :param precomputed: indicates whether or not the MFR and VS values are already computed or \
        be computed on the go
        :param test_nr: which number of test folder should be loaded (generally 0)
        :param split: used if the plot has a split to deal with akward value differences
        :param refvals: reference values used for VS plots
        :param includePhaseLock: indicates whether an example of a phase-locked responde should be plotted
    """

    if includePhaseLock and split:
        raise ValueError("The split option is not available for plots that \
                          include spike histograms (i.e. includePhaseLock = True)")
    if includePhaseLock:
        fig, ax = plt.subplots(3, 1, figsize=(5, 8))
    else:
        if not split:
            fig, ax = plt.subplots(2, 1, figsize=(4, 8))
        else:
            fig = plt.figure(figsize=(4, 8))
            ylim, ylim2, yticks_0, yticks_1 = split
            ylimratio = (ylim[1] - ylim[0])/(ylim2[1] - ylim2[0] + ylim[1] - ylim[0])
            ylim2ratio = 1 - ylimratio
            gs = gridspec.GridSpec(4, 1, height_ratios=[ylimratio, ylim2ratio, 0.2, 1.0])
            ax = [0, 0, 0]
            ax[-1] = fig.add_subplot(gs[0])
            ax[0] = fig.add_subplot(gs[1])
            ax[1] = fig.add_subplot(gs[3])

    if not test_nr:
        test_nr = np.zeros(len(paths)).astype(int)

    for i, path in enumerate(paths):

        itdoptions, mfr, vs, t = loadMeasurements(path, test_nr[i], precomputed)
        plotMeanAndErr(itdoptions, mfr, ax[0], colors[i % len(colors)], alpha=0.3)

        if split:
            plotMeanAndErr(itdoptions, mfr, ax[2], colors[i % len(colors)], alpha=0.3)
        plotMeanAndErr(itdoptions, vs, ax[1], colors[i % len(colors)], alpha=0.2)

        if refvals:
            ax[1].plot(itdoptions, [refvals[i]]*len(itdoptions), ls='--', color='gray')

        if includePhaseLock:
            print "Loading spikes for path {}... (this may take a while)".format(i)
            v_spikes, T, dt = loadSpikeTrain(paths[i], test_nr[i])
            ax[2].hist(np.where(v_spikes[0][:200000] > 0)[0]*dt % T, T/dt, range=[0, T], edgecolor=colors[i % len(colors)], color=colors[i % len(colors)])

    if not split:
        ax[0].set_ylabel('Mean firing rate')
    ax[1].set_ylabel('Vector strength')
    ax[0].set_xlabel('ITD (ms)')
    ax[1].set_xlabel('ITD (ms)')
    ax[1].set_ylim([0, 1])
    cleanAxes(ax)
    maxTicks(ax, (3, 4))

    if split:
        ax[-1].set_ylabel('Mean firing rate', position=(1.0, 0.2))
        ax[-1].set_ylim(ylim)
        ax[0].set_ylim(ylim2)
        ax[-1].yaxis.set_major_locator(plt.MaxNLocator(5))
        ax[0].yaxis.set_major_locator(plt.MaxNLocator(3))
        ax[-1].spines["bottom"].set_visible(False)
        ax[-1].set_xticks([])
        ax[0].yaxis.set_label_coords(0.05, 0.5, transform=fig.transFigure)

        # Add split icons
        kwargs = dict(color='k', clip_on=False, lw=2.0)
        xlim = ax[0].get_xlim()
        dx = .05*(xlim[1]-xlim[0])
        dy = .01*(ylim[1]-ylim[0])/ylimratio
        ax[-1].plot((xlim[0] - dx, xlim[0] + dx), (ylim[0] - dy - 30, ylim[0] + dy - 30), **kwargs)
        dy = .01*(ylim2[1]-ylim2[0])/ylim2ratio
        ax[0].plot((xlim[0] - dx, xlim[0] + dx), (ylim2[1] - dy + 20, ylim2[1] + dy + 20), **kwargs)
        ax[-1].set_yticks(yticks_0)
        ax[0].set_yticks(yticks_1)
        ax[-1].set_xlim(xlim)
        ax[0].set_xlim(xlim)

    ax[0].set_xlim(min(itdoptions), max(itdoptions))
    ax[1].set_xlim(min(itdoptions), max(itdoptions))
    ax[0].set_ylim(0, ax[0].get_ylim()[1])

    if includePhaseLock:
        ax[2].set_ylabel('Spikes / bin')
        x_tick = np.linspace(0, T, 3)
        x_label = [r"$0$", r"$\pi$", r"$2\pi$"]
        ax[2].set_xticks(x_tick)
        ax[2].set_xticklabels(x_label)
        ax[2].set_xlabel(r'Phase $\phi$')
        ax[2].set_xlim([0, T])

    plt.tight_layout()
    if split:
        gs.update(hspace=0.2)


def loadMeasurementsPerItd(paths, ids, test_nr):

    """
        Loads the test results for a given set of paths and indicates which test folders
        should be loaded (in test_nr). The ids is a list that indicates to which class
        each path belongs. I.e. if two simulations had the same parameter settings they
        are concatenated.
    """

    ncat = max(ids) + 1
    itdoptions, mfr, vs, t = loadMeasurements(paths[0], test_nr[0], precomputed=True)
    mod_per_itd = [[[] for x in range(len(itdoptions))] for y in range(ncat)]
    vs_per_itd = [[[] for x in range(len(itdoptions))] for y in range(ncat)]
    modpre_per_itd = [[[] for x in range(len(itdoptions))] for y in range(ncat)]
    vspre_per_itd = [[[] for x in range(len(itdoptions))] for y in range(ncat)]
    nr_per_itd = np.zeros((ncat, len(itdoptions)))
    itdoptions_container = [0]*ncat

    for i, path in enumerate(paths):

        itdoptions, mfr, vs, t = loadMeasurements(path, test_nr[i], precomputed=True)

        # Load weights
        with gzip.open('{}/data_per_batch/{}_w.json.gz'.format(path, t - 1), 'r') as f:
            w = np.array(json.loads(f.readlines()[0]))

        # Sort mean firing rates and vector strengths on BITD
        mfr = np.array(mfr).T
        maxMFRs = np.argmax(mfr, axis=1)
        mfr = mfr / np.max(mfr, axis=1)[:, np.newaxis]
        sortedMFR = mfr[np.argsort(maxMFRs)]
        sortedVS = np.array(vs).T[np.argsort(maxMFRs)]

        for x in range(sortedMFR.shape[0]):
            mod = (np.max(sortedMFR[x]) - np.min(sortedMFR[x]))/np.max(sortedMFR[x])
            bitd = np.argmax(sortedMFR[x])
            modpre_per_itd[ids[i]][bitd].append(mod)
            vspre_per_itd[ids[i]][bitd].append(sortedVS[x])

        # Remove monaural neurons from data
        left_idx = w.shape[1]/2
        leftweights = np.where(np.sum(w[:, left_idx:], axis=1) > 0)
        rightweights = np.where(np.sum(w[:, :left_idx], axis=1) > 0)
        bothears = np.intersect1d(leftweights, rightweights)
        bothearMFR = mfr[bothears]
        maxMFRs = np.argmax(bothearMFR, axis=1)
        sortedMFR = bothearMFR[np.argsort(maxMFRs)]

        vs = np.array(vs).T
        bothearVS = vs[bothears]
        sortedVS = bothearVS[np.argsort(maxMFRs)]

        for x in range(sortedMFR.shape[0]):
            mod = (np.max(sortedMFR[x]) - np.min(sortedMFR[x]))/np.max(sortedMFR[x])
            bitd = np.argmax(sortedMFR[x])
            mod_per_itd[ids[i]][bitd].append(mod)
            vs_per_itd[ids[i]][bitd].append(sortedVS[x])
            nr_per_itd[ids[i], bitd] += 1

        itdoptions_container[ids[i]] = itdoptions

    return modpre_per_itd, vspre_per_itd, mod_per_itd, vs_per_itd, nr_per_itd, itdoptions_container


def plotModulationOrVectorstrength(mod_per_itd, modpre_per_itd, labels, ptype, colors):

    """
        Plots a barplot of a range of values, a left plot (modpre_per_itd) and right plot (mod_per_itd)
        for each set.

        :param ptype: Stringt of what should be plotted on the y-axis
    """

    fig, ax = plt.subplots(1, figsize=(5, 3))
    modflattened = [[item for sublist in l for item in sublist] for l in mod_per_itd]
    modpreflattened = [[item for sublist in l for item in sublist] for l in modpre_per_itd]
    ncat = len(modflattened)
    for i in range(ncat):
        plotBar(np.array([i]), [np.array(modflattened[i])], ax, colors[i % len(colors)], width=0.4)
        if i != ncat - 1:
            plotBar(np.array([i]) - 0.45, [np.array(modpreflattened[i])], ax, color_variant(colors[i % len(colors)], 50), width=0.4)

    ax.set_ylim([0, 1])
    ax.set_xticks([])

    cleanAxes([ax])
    ax.set_ylabel(ptype)

    xticks = np.array(range(len(labels))) - 0.2
    xticks[-1] += 0.2
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)


def plotDistribution(labels, colors, nr_per_itd, itdoptions, width=0.2, zero=True):
    """
        Plots a barplot of a histogram list (nr_per_itd) on x-axis value itdoptions
    """

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.0))
    if zero:
        r = len(labels)
    else:
        r = len(labels) - 1
    for i in range(r):
        plotBar(np.array(itdoptions[i]), np.array(nr_per_itd[i]) / np.sum(nr_per_itd[i])*100, ax[i], colors[i % len(colors)], label=labels[i], width=width)
        ax[i].set_xlim([min(itdoptions[i]), max(itdoptions[i])])
        ax[i].set_title(labels[i])
        ax[i].set_ylim([0, 100])
        ax[i].set_xlabel('Best ITD (ms)')
        ax[i].set_title(labels[i], y=0.85, x=0.5)

    cleanAxes(ax)
    if min(itdoptions[i]) > -1:
        maxTicks(ax, (3, 5))
    else:
        maxTicks(ax, (4, 5))
    ax[0].set_ylabel('% MSO neurons')

    plt.tight_layout()


def plotDelays(path, t, i):
    """
        Plots the delay distribution of neuron i at time t for a simulation stored in path
    """

    colors = basiccolors()

    fig, ax = plt.subplots(2, 1, figsize=(5, 7), sharey=True)
    if t == 0:
        w = np.ones((1, 500))
        dt = 0.005
        T = 1.0
    else:
        with gzip.open('{}/data_per_batch/{}_w.json.gz'.format(path, int(t - 1)), 'r') as f:
            w = np.array(json.loads(f.readlines()[0]))
        sim = MSOModel.loadLearnedModel(path, t - 1)
        dt = sim.dt
        T = sim.T

    with open('{}/delays.txt'.format(path, int(t - 1)), 'r') as f:
        delays = np.array(json.loads(f.readlines()[0]))

    left_idx = delays.shape[1]/2
    plt.cla()

    ax[0].hist(delays[i, left_idx:], T/dt, range=(4, 10),
               weights=w[i, left_idx:], color=colors[0], edgecolor=colors[0])
    ax[0].set_ylabel("Weights $w$ / bin")
    ax[0].set_title('Left ear')

    ax[1].hist(delays[i, :left_idx], T/dt, range=(4, 10),
               weights=w[i, :left_idx], color=colors[0], edgecolor=colors[0])
    ax[1].set_xlabel('Delay $d$ (ms)')
    ax[1].set_ylabel("Weights $w$ / bin")
    ax[1].set_title('Right ear')

    cleanAxes(ax)

def plotMeanFiringRateImshow(path, label, rvb, test_nr):

    fig, ax = plt.subplots(2, figsize=(5, 10))

    with open('{}/simulation_settings.txt'.format(path), 'r') as f:
        true = True
        sim = eval(f.readlines()[0])

    dt = sim['dt']
    T = sim['T']

    directory = [f for f in listdir(path + '/tests') if isdir('{}/tests/{}'.format(path, f))][test_nr]
    
    with open('{}/tests/{}/parameters.txt'.format(path, directory), 'r') as f:
        res = json.loads(f.readlines()[0])
        itdoptions = res['ITDoptions']
        ts = res['t']
    
    with gzip.open('{}/data_per_batch/{}_w.json.gz'.format(path, ts - 1), 'r') as f:
        w = np.array(json.loads(f.readlines()[0]))

    path = '{}/tests'.format(path)

    mfr = [0]*len(itdoptions)
    vs = [0]*len(itdoptions)

    for j, itd in enumerate(itdoptions):

        npath = path + '/{}/mfr_{}.json.gz'.format(directory, itd)
        with gzip.open(npath, 'r') as f:
            mfr[j] = np.array(json.loads(f.readlines()[0]))

        npath = path + '/{}/vs_{}.json.gz'.format(directory, itd)
        with gzip.open(npath, 'r') as f:
            vs[j] = np.array(json.loads(f.readlines()[0]))

    if abs(itdoptions[0]) > abs(itdoptions[-1]):
        mfr.append(mfr[0])
        vs.append(vs[0])
        itdoptions = np.append(itdoptions, -itdoptions[0])
    else:
        mfr.insert(0, mfr[-1])
        vs.insert(0, vs[-1])
        itdoptions = np.insert(itdoptions, 0, -itdoptions[-1])

    for j in range(len(ax)):
        ax[j].spines["right"].set_visible(False)
        ax[j].spines["top"].set_visible(False)
        ax[j].get_xaxis().tick_bottom()
        ax[j].get_yaxis().tick_left()
        ax[j].tick_params('both', length=5, width=2, which='major')
        ax[j].tick_params('both', length=5, width=1, which='minor')
        ax[j].set_xlabel('ITD (ms)')

        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks)
        xloc = plt.MaxNLocator(max_yticks)
        ax[j].yaxis.set_major_locator(yloc)
        ax[j].xaxis.set_major_locator(xloc)

    mfr = np.array(mfr).T
    maxMFRs = np.argmax(mfr, axis=1)
    mfr = mfr / np.max(mfr, axis=1)[:, np.newaxis]
    sortedMFR = mfr[np.argsort(maxMFRs)]
    stepsize = abs(itdoptions[0] - itdoptions[1])/2.0
    extent = [np.min(itdoptions) - stepsize, np.max(itdoptions) + stepsize, 0, sortedMFR.shape[0]]
    xticks = np.linspace(min(itdoptions), max(itdoptions), 5)

    ax[0].imshow(sortedMFR, vmin=0, vmax=1, interpolation='nearest', extent=extent, aspect='auto', cmap=rvb)
    ax[0].set_xlabel('ITD')
    ax[0].set_xticks(xticks)
    ax[0].set_ylabel('Neuron no.')
    
    left_idx = w.shape[1]/2
    leftweights = np.where(np.sum(w[:, left_idx:], axis=1) > 0)
    rightweights = np.where(np.sum(w[:, :left_idx], axis=1) > 0)
    bothears = np.intersect1d(leftweights, rightweights)
    bothearMFR = mfr[bothears]
    maxMFRs = np.argmax(bothearMFR, axis=1)
    sortedMFR = bothearMFR[np.argsort(maxMFRs)]
    extent = [np.min(itdoptions) - stepsize, np.max(itdoptions) + stepsize, 0, sortedMFR.shape[0]]

    im = ax[1].imshow(sortedMFR, vmin=0, vmax=1, interpolation='nearest', extent=extent, aspect='auto', cmap=rvb)
    ax[1].set_xlabel('ITD')
    ax[1].set_xticks(xticks)
    ax[1].set_ylabel('Neuron no.')

    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.25)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.25)
    plt.colorbar(im, cax=cax1)
    plt.colorbar(im, cax=cax2)
    plt.tight_layout()
