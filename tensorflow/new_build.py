import glob
import random
import re
from collections import Counter
import numpy as np

class DocReader():
    def __init__(self):
        pass

    def create_bag_of_words(self,filePaths):
        '''
        Input:
          filePaths: Array. A list of absolute filepaths
        Returns:
          bagOfWords: Array. All tokens in files
        '''
        bagOfWords = []
        # regex = re.compile("\w+")
        for filePath in filePaths:
            print filePath
            with open(filePath) as f:
                raw = f.read()
                # raw = re.sub(regex,'',raw)
                tokens = re.findall("\w+", raw)
                # tokens = raw.split()
                for token in tokens:
                    bagOfWords.append(token)
        return bagOfWords

    def get_feature_matrix(self,filePaths, featureDict):
        '''
        create feature/x matrix from multiple text files
        rows = files, cols = features
        '''
        featureMatrix = np.zeros(shape=(len(filePaths),
                                          len(featureDict)),
                                   dtype=float)
        # regex = re.compile("\w+")
        for i,filePath in enumerate(filePaths):
            with open(filePath) as f:
                raw = f.read()
                # raw = re.sub(regex,'',_raw)
                # tokens = raw.split()
                tokens = re.findall("\w+", raw)
                fileUniDist = Counter(tokens)
                for key,value in fileUniDist.items():
                    if key in featureDict:
                        featureMatrix[i,featureDict[key]] = value
        return featureMatrix

    def regularize_vectors(self,featureMatrix):
        '''
        Input:
          featureMatrix: matrix, where docs are rows and features are columns
        Returns:
          featureMatrix: matrix, updated by dividing each feature value by the total
          number of features for a given document
        '''
        for doc in range(featureMatrix.shape[0]):
            totalWords = np.sum(featureMatrix[doc,:],axis=0)
            featureMatrix[doc,:] = np.multiply(featureMatrix[doc,:],(1/totalWords))
        return featureMatrix

    def input_data(self,publicDir,internalDir,restrictedDir, highlyRestrictedDir, percentTest,cutoff):
        ''' 
        Input:
          hamDir: String. dir of ham text files
          spamDir: String. dir of spam text file
          percentTest: Float. percentage of all data to be assigned to testset
        Returns:
          trainPaths: Array. Absolute paths to training emails
          trainY: Array. Training labels, 0 or 1 int.
          testPaths: Array. Absolute paths to testing emails
          testY: Array. Testing labels, 0 or 1 int.
        '''
        pathLabelPairs={}
        for publicPath in glob.glob(publicDir+'*'):
            pathLabelPairs.update({publicPath:(1,0,0,0)})
        for internalPath in glob.glob(internalDir+'*'):
            pathLabelPairs.update({internalPath:(0,1,0,0)})
        for restrictedPath in glob.glob(restrictedDir+'*'):
            pathLabelPairs.update({restrictedPath:(0,0,1,0)})
        for highlyRestrictedPath in glob.glob(highlyRestrictedDir+'*'):
            pathLabelPairs.update({highlyRestrictedPath:(0,0,0,1)})

        # get test set as random subsample of all data
        numTest = int(percentTest * len(pathLabelPairs))
        testing = set(random.sample(pathLabelPairs.items(),numTest))

        # delete testing data from superset of all data
        for entry in testing:
            del pathLabelPairs[entry[0]]

        # split training tuples of (path,label) into separate lists
        trainPaths=[]
        trainY=[]
        for item in pathLabelPairs.items():
            trainPaths.append(item[0])
            trainY.append(item[1])

        # split testing tuples of (path,label) into separate lists
        testPaths=[]
        testY=[]
        for item in testing:
            testPaths.append(item[0])
            testY.append(item[1])

        # create feature dictionary of n-grams
        bagOfWords = self.create_bag_of_words(trainPaths)

        # throw out low freq words
        freqDist = Counter(bagOfWords)
        newBagOfWords=[]
        for word,freq in freqDist.items():
            if freq > cutoff:
                newBagOfWords.append(word)
        features = set(newBagOfWords)
        featureDict = {feature:i for i,feature in enumerate(features)}

        # make feature matrices
        trainX = self.get_feature_matrix(trainPaths,featureDict)
        testX = self.get_feature_matrix(testPaths,featureDict)

        # regularize length
        trainX = self.regularize_vectors(trainX)
        testX = self.regularize_vectors(testX)

        # cast as ndarrays
        trainY = np.asarray(trainY)
        testY = np.asarray(testY)

        return trainX, trainY, testX, testY


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ham','--hamDir')
    parser.add_argument('-spam','--spamDir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #import sys, argparse
    # get user input
    #args = parse_user_args()
    #hamDir = args.hamDir
    #spamDir= args.spamDir
    publicDir="source_files/public/"
    internalDir="source_files/internal/"
    restrictedDir="source_files/restricted/"
    highlyRestrictedDir="source_files/highly_restricted/"

    reader = DocReader()
    
    trainX,trainY,testX,testY = reader.input_data(publicDir=publicDir,
                                                  internalDir=internalDir,
                                                  restrictedDir=restrictedDir,
                                                  highlyRestrictedDir=highlyRestrictedDir,
                                                  percentTest=.1,
                                                  cutoff=15)

    print(trainX.shape)
    print(trainY.shape)    
    print(testX.shape)
    print(testY.shape)    

    np.savetxt("trainX.csv", trainX, delimiter="\t")
    np.savetxt("trainY.csv", trainY, delimiter="\t")
    np.savetxt("testX.csv", testX, delimiter="\t")
    np.savetxt("testY.csv", testY, delimiter="\t")

    print(trainX[:10,:])
    print(trainY[:10,:])
