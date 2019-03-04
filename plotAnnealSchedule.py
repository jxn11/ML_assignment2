import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

showPlotOne = True

if showPlotOne:
    tempVals = []
    tempVals.append(0.5)

    iterStop = 0

    for i in range(1000):
        tempVals.append(tempVals[-1]*.8)

        if tempVals[-1] < 0.000001 and iterStop == 0:
            iterStop = i

    plt.plot(list(range(len(tempVals))), tempVals)
    plt.axvline(x=iterStop, color='red')
    plt.legend(['Annealing Schedule', 'Stop Condition Met'])
    plt.xlabel('Iterations')
    plt.ylabel('Temperature')
    plt.title('Optimal Hyperparameter Search Values')
    plt.show()
hyperTemps = tempVals

# look for better schedule
tempVals = np.ones((5, 1000)) * 100

iterStops = np.zeros(5)

coolRates = [0.8, 0.9, 0.93, 0.96, 0.98]

for i in range(5):
    for j in range(1,1000):

        tempVals[i,j] = tempVals[i,j-1] * coolRates[i]

        if tempVals[i,j] < 0.000001 and iterStops[i] < 1:
            iterStops[i] = j

print(iterStops)
print(tempVals[-1,-1])

cmap = cm.Paired
testColors = [cmap(1), cmap(3), cmap(5), cmap(7), cmap(9)]
trainColors = [cmap(0), cmap(2), cmap(4), cmap(6), cmap(8)]

plt.figure()
for i in range(tempVals.shape[0]):
    plt.plot(list(range(tempVals.shape[1])), tempVals[i,:], color=testColors[i])
    if i < 7:
        plt.axvline(x=iterStops[i], color=trainColors[i])
plt.xlabel('Iterations')
plt.ylabel('Temperature')
plt.title('Simulated Annealing:\nExponential Temperature Decay Schedules')
plt.legend(['Decay Rate = 0.8', '0.8 Stop Threshold', \
            'Decay Rate = 0.9', '0.9 Stop Threshold', \
            'Decay Rate = 0.93', '0.93 Stop Threshold', \
            'Decay Rate = 0.96', '0.96 Stop Threshold', \
            'Decay Rate = 0.98', '0.98 Stop Threshold', \
            ])

plt.show()


f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(list(range(len(hyperTemps))), hyperTemps)
ax1.axvline(x=iterStop, color='red')
ax1.legend(['Annealing Schedule', 'Stop Condition Met'])
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Temperature')
ax1.set_title('Optimal Hyperparameter Search Values')

for i in range(tempVals.shape[0]):
    ax2.plot(list(range(tempVals.shape[1])), tempVals[i,:], color=testColors[i])
    if i < 7:
        ax2.axvline(x=iterStops[i], color=trainColors[i])
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Temperature')
ax2.set_title('Simulated Annealing:\nExponential Temperature Decay Schedules')
ax2.legend(['Decay Rate = 0.8', '0.8 Stop Threshold', \
            'Decay Rate = 0.9', '0.9 Stop Threshold', \
            'Decay Rate = 0.93', '0.93 Stop Threshold', \
            'Decay Rate = 0.96', '0.96 Stop Threshold', \
            'Decay Rate = 0.98', '0.98 Stop Threshold', \
            ])

plt.show()


# ax1.imshow(pop_xRate, vmin=minVal, vmax=maxVal, interpolation='nearest', \
#            cmap=plt.cm.hot)
# ax1.set_ylabel('Initial Population Size')
# ax1.set_yticks(np.arange(len(popSizes)))
# ax1.set_yticklabels(popSizes)
# ax1.set_xlabel('Crossover Rate')
# ax1.set_xticks(np.arange(len(xRates)))
# ax1.set_xticklabels(xRates)
