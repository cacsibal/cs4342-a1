import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

"""
calculates the percent of guesses correct

yhat: predicted guess vector
y: ground truth vector

returns a float between [0, 1] representing the percentage of correct guesses
"""
def fPC (y, yhat) -> float:
    return sum(1 for true, pred in zip(y, yhat) if true == pred) / len(y)

"""
idk what this does yet

predictors: a list of predictors in the form (r1, c1, r2, c2)
X: a 3D array
y: ground truth vector

returns 
"""
def measureAccuracyOfPredictors (predictors, X, y) -> list[float]:
    return [fPC(y, predictors(image)) for image in X]

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    show = False
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Display the merged result
        plt.show()

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
