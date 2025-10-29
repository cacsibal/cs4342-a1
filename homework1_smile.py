import random

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

"""
calculates the percent of guesses correct

yhat: predicted guess vector
y: ground truth vector

returns a float between [0, 1] representing the percentage of correct guesses
"""
def fPC (y, yhat):
    return np.sum(y == yhat) / len(y)

def vectorize_images(X):
    n, m, _ = X.shape
    return X.reshape(n, m * m).T

def get_flattened_index(coord):
    i, j = coord
    return i * 24 + j

"""
idk what this does yet

predictors: a list of predictors in the form (r1, c1, r2, c2)
X: a 3D array
y: ground truth vector

returns 
"""
def measureAccuracyOfPredictors (predictors, X, y):
    accuracies = []

    for i, g_j in enumerate(predictors):
        if i % 10000 == 0:
            print(f"Done with {i} predictors")
        predictions = np.array([g_j(x) for x in X])
        accuracies.append(fPC(y, predictions))

    return accuracies

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

def make_predictor(r1, c1, r2, c2):
    return lambda x: 1 if x[r1, c1] > x[r2, c2] else 0

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    predictors = []

    for r1 in range(24):
        for c1 in range(24):
            for r2 in range(24):
                for c2 in range(24):
                    predictors.append(make_predictor(r1, c1, r2, c2))

    # for _ in range(10000):
    #     r1, c1 = random.randint(0, 23), random.randint(0, 23)
    #     r2, c2 = random.randint(0, 23), random.randint(0, 23)
    #     if not (r1 == r2 and c1 == c2):
    #         predictors.append(make_predictor(r1, c1, r2, c2))

    print(max(measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)))

    # print(testingFaces.shape)
    # print(vectorize_images(testingFaces).shape)
    #
    # print(testingFaces[0][5][10])
    # print(vectorize_images(testingFaces)[5 * 24 + 10][0])
    #
    # print(trainingLabels[0:9])
