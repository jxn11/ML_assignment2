import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pickle

showHC_allIts = False

# load in the learning curve data
with open('./full_Iters_bkup/hillClimbFullIterResults_v3_plots.pkl', 'rb') as f:
    hillClimbData = pickle.load(f)
hillClimbCurve = hillClimbData[0]
hillClimbVal = hillClimbData[1]
hillClimbTest = hillClimbData[2]

with open('./full_Iters_bkup/simAnnealFullIterResults_v5_temp100_decay98_plots.pkl', 'rb') as f:
    annealData = pickle.load(f)
annealCurve = annealData[0]
annealVal = annealData[1]
annealTest = annealData[2]

with open('./full_Iters_bkup/genAlgFullIterResults.pkl', 'rb') as f:
    genAlgData = pickle.load(f)
genAlgCurve = genAlgData[0]
genAlgVal = genAlgData[1]
genAlgTest = genAlgData[2]

if showHC_allIts:

    #randRestart mean
    meanScore = np.mean(hillClimbCurve[:,-1])
    stdScore = np.std(hillClimbCurve[:,-1])

    # plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(hillClimbCurve.shape[0]):
        ax1.plot(list(range(1, hillClimbCurve.shape[1]+1)), hillClimbCurve[i,:])
    ax1.set_title('Random Hillclimb Training Curves (All Restarts)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Classificiation Accuracy')

    ax2.hist(hillClimbCurve[:,-1], color='c', edgecolor='k', alpha=0.65)
    ax2.axvline(meanScore, color='k', linestyle='dashed', linewidth=1)
    ax2.set_title('Random Hillclimb Outcome Distribution\n(mean score: {0:2.2f}, standard deviaiton: {1:2.2f})'.format(meanScore, stdScore))
    ax2.set_xlabel('Classificaiton Accuracy')
    ax2.set_ylabel('Number of Observations')
    # plt.plot(list(range(1, len(annealCurve)+1)), annealCurve)
    # plt.plot(list(range(1, len(genAlgCurve)+1)), genAlgCurve)
    plt.show()

# create time vectors
climbTime = 0.021
climbTime2 = 0.315
hillClimbIters = list(range(1, hillClimbCurve.shape[1]+1))
hc_time = [x * climbTime for x in hillClimbIters]
hc_time2 = [x * climbTime2 for x in hillClimbIters]

simTime = 0.02
simIters = list(range(1, len(annealCurve)+1))
sim_time = [x * simTime for x in simIters]

genTime = 1.16
genIters = list(range(1, len(genAlgCurve)+1))
gen_time = [x * genTime for x in genIters]

cmap = cm.Paired
testColors = [cmap(1), cmap(3), cmap(5), cmap(7), cmap(9)]
trainColors = [cmap(0), cmap(2), cmap(4), cmap(6), cmap(8)]

#plot
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(list(range(1, hillClimbCurve.shape[1]+1)), hillClimbCurve[12,:], color=testColors[0],linewidth=2)
ax1.plot(list(range(1, hillClimbCurve.shape[1]+1)), np.mean(hillClimbCurve,0), color=trainColors[0],linewidth=2)
ax1.plot(list(range(1, len(annealCurve)+1)), annealCurve, color=testColors[1],linewidth=2)
ax1.plot(list(range(1, len(genAlgCurve)+1)), genAlgCurve, color=testColors[2],linewidth=2)
ax1.set_title('Classification Over Training Iterations Comparison')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Classificiation Accuracy')
ax1.legend(['Randomized Hillclimbing (best iteration)', 'Randomized Hillclimbing (average)','Simulated Annealing', 'Genetic Algorithm'])

ax2.plot(hc_time2, hillClimbCurve[12,:], color=testColors[0],linewidth=2)
ax2.plot(hc_time, hillClimbCurve[12,:], color=trainColors[0],linewidth=2)
ax2.plot(sim_time, annealCurve, color=testColors[1],linewidth=2)
ax2.plot(gen_time, genAlgCurve, color=testColors[2],linewidth=2)
ax2.set_title('Classification Over Training Time Comparison')
ax2.set_xlabel('Time (minutes)')
# ax2.set_ylabel('Classificiation Accuracy')
ax2.legend(['Randomized Hillclimbing', 'Randomized Hillclimbing (single iteration)', 'Simulated Annealing', 'Genetic Algorithm'])
plt.suptitle('Optimization Technique Comparion')
plt.show()
