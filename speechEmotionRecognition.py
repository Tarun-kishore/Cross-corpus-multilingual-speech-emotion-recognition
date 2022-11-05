from __future__ import division, print_function
import os,glob
import numpy as np 
import librosa
import parselmouth
import tsfel
import torchaudio
import torch
from parselmouth.praat import call
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
#from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC
import opensmile


def extractFeatures(fileName):
    y, sr = librosa.load(fileName)

    result=np.array([])

    # extracting chroma features
    stft=np.abs(librosa.stft(y))
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    result=np.hstack((result, chroma))

    # extracting mfcc features
    mfccs=np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)[:20]
    result=np.hstack((result, mfccs))

    # extracting rolloff features
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99).T,axis=0)
    result=np.hstack((result, rolloff))

    # extracting centroid features
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T,axis=0)
    result=np.hstack((result, centroid))

    # extracting bandwidth features
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T,axis=0)
    result=np.hstack((result, bandwidth))

    # extracting rms features
    rms = np.mean(librosa.feature.rms(y=y).T,axis=0)
    result=np.hstack((result, rms))

    # extracting fundamental frequency
    # f0=librosa.pyin(y=y, fmin=librosa.note_to_hz('c2'), fmax=librosa.note_to_hz('c7'))
    # f0=np.array(f0)

    # f0 = np.mean(f0.T,axis=0)
    # result=np.hstack((result, f0))


    # extracting contrast features
    S = np.abs(librosa.stft(y))
    contrast = np.mean(librosa.feature.spectral_contrast(S=S, sr=sr).T,axis=0)
    result=np.hstack((result, contrast))

    # extracting zero crossing rate features
    zeroCrossingRate=np.mean(librosa.feature.zero_crossing_rate(y=y).T,axis=0)
    result=np.hstack((result, zeroCrossingRate))

    # extra eGeMAPS features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    eGeMAPS=smile.process_signal(y,sr)
    eGeMAPS = np.mean(eGeMAPS,axis=0)
    result=np.hstack((result, eGeMAPS))

    print(fileName)
    return result


def loadData(folderName, emotionsConditions):
    x,y = [] , []
    num=250
    for folder in glob.glob(folderName):
        for file in glob.glob(folder+'/*.wav'):
            num=num-1
            if(num<=0):
                break
            fileName = os.path.basename(file)
            emotion = emotionsConditions(fileName)
            feature = extractFeatures(file)
            x.append(feature)
            y.append(emotion)


    preprocessor = SimpleImputer(missing_values=np.nan, strategy='mean')
    preprocessor.fit(x)
    x = preprocessor.transform(x)
    sm = SMOTE(k_neighbors=3,random_state=42)
    x, y = sm.fit_resample(x, y)
    x=x-np.mean(x)
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return x,y




def saveeConditions(x):
    if(x[0]=='s'):
        if(x[1]=='a'):
            return 'sadness'
        else:
            return 'surprised'
    elif(x[0]=='a'): 
        return 'angry'
    elif(x[0]=='d'): 
        return 'disgust'
    elif(x[0]=='f'): 
        return 'fear'
    elif(x[0]=='h'): 
        return 'happiness'
    elif(x[0]=='n'): 
        return 'neutral'


saveeX,saveeY=loadData('data/AudioData/*',saveeConditions)

def urduConditions(name):
    x=name.split('_')[2] 
    if(x[0]=='A'): 
        return 'angry'
    elif(x[0]=='N'): 
        return 'neutral'
    elif(x[0]=='H'): 
        return 'happiness'
    elif(x[0]=='S'): 
        return 'sadness'


urduX,urduY=loadData('data/Urdu/*',urduConditions)


def emodbConditions(x):
    if(x[5]=='A' or x[5]=='W'):
        return 'angry'
    elif(x[5]=='B' or x[5]=='L'): 
        return 'boredom'
    elif(x[5]=='D' or x[5]=='E'): 
        return 'disgust'
    elif(x[5]=='F' or x[5]=='A'): 
        return 'fear'
    elif(x[5]=='H' or x[5]=='F'): 
        return 'happiness'
    elif(x[5]=='S' or x[5]=='T'): 
        return 'sadness'
    elif(x[5]=='N'):
        return 'neutral'

# emodbX,emodbY=loadData('data/emo-db/*',emodbConditions)

def emovoConditions(name):
    x=name.split('-')[0] 
    if(x=="rab"): 
        return 'angry'
    elif(x=="neu"): 
        return 'neutral'
    elif(x=="gio"): 
        return 'happiness'
    elif(x=="tri"): 
        return 'sadness'
    elif(x=="pau"): 
        return 'fear'
    elif(x=="sor"): 
        return 'surprise'
    elif(x=="dis"): 
        return 'disgust'

# emovoX,emovoY=loadData('data/emovo/EMOVO/*',emovoConditions)

def cal_accuracy(y_test, y_pred):
      
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))


class SVM():
    """
        Simple implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm for training.
    """

    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001):
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):
        # Initialization
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        kernel = self.kernels[self.kernel_type]
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = self.get_rnd_int(0, n-1, j)  # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i, :], X[j, :], y[i], y[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - \
                    2 * kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(
                    self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" %
                      (self.max_iter))
                return
        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count

    def predict(self, X):
        return self.h(X, self.w, self.b)

    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha, y))
    # Prediction

    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)
    # Prediction error

    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

    def get_rnd_int(self, a, b, z):
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = rnd.randint(a, b)
            cnt = cnt+1
        return i
    # Define kernels

    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)


def SMO(x_train, x_test, y_train, y_test):
    #print(y_train)
    # clf.fit(x_train, y_train)
    # pred_lin = clf.score(x_test, y_test)
    # print("Test Accuracy: ", pred_lin*100)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    svm_model_linear = SVC(kernel = 'poly', C = 0.1).fit(x_train, y_train) 
    svm_predictions = svm_model_linear.predict(x_test) 
    print("Results Using Support Vector Machine: ")
    cal_accuracy(y_test, svm_predictions)
    return svm_model_linear
    # creating a confusion matrix 
    #print(confusion_matrix(y_test, svm_predictions) )
    # # print(y_train)
    # y_train_int = []
    # for y in y_train:
    #     if y == 'angry':
    #         y_train_int.append(1)
    #     if y == 'happiness':
    #         y_train_int.append(2)
    #     if y == 'neutral':
    #         y_train_int.append(3)
    #     if y == 'sadness':
    #         y_train_int.append(4)
    # y_test_int = []
    # for y in y_test:
    #     if y == 'angry':
    #         y_test_int.append(1)
    #     if y == 'happiness':
    #         y_test_int.append(2)
    #     if y == 'neutral':
    #         y_test_int.append(3)
    #     if y == 'sadness':
    #         y_test_int.append(4)
    # d = dict([(y,x+1) for x,y in enumerate(sorted(set(y_train)))])
    # d2 = dict([(y,x+1) for x,y in enumerate(sorted(set(y_test)))])
    # model = SVM()
    # support_vectors, iterations = model.fit(x_train, y_train_int)

    # Support vector count
    # sv_count = support_vectors.shape[0]

    # Make prediction
    # y_hat = model.predict(x_test)
    # print("Results Using SMO")
    # cal_accuracy(y_test_int, y_hat)



def decisionTree(x_train,x_test,y_train,y_test):
    
    # selector= SelectKBest(mutual_info_classif, k=28)
    # x_train= selector.fit_transform(x_train,y_train)
    # x_test=selector.transform(x_test)
    clf = DecisionTreeClassifier(
                criterion = "entropy", random_state = 100,
                max_depth = 15, min_samples_leaf = 5)
      
    clf_return = clf
    clf.fit(x_train, y_train)
      
    y_pred = clf.predict(x_test)
    
    print("Results Using J48:")
    cal_accuracy(y_test, y_pred)

    return clf_return 

def random_forest(x_train, x_test, y_train, y_test):
    # selector= SelectKBest(mutual_info_classif, k=38)
    # x_train= selector.fit_transform(x_train,y_train)
    # x_test=selector.transform(x_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=100)
    clf_return = model
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Results Using random forest:")
    cal_accuracy(y_test, y_pred)

    return model

def ensemble(x_train,x_test,y_train,y_test,dt_clf,rf_clf,smo_clf):
    clf = VotingClassifier(
        estimators=[('dt', dt_clf), ('smo', smo_clf), ('rf', rf_clf)],
        voting='hard'
    )
      
    clf_return = clf
    clf.fit(x_train, y_train)
      
    y_pred = clf.predict(x_test)
    
    print("Results Using ensemble:")
    cal_accuracy(y_test, y_pred)

    return clf_return 


def classify(x_train_pre,x_test_pre,y_train_pre,y_test_pre):

    length = len(x_test_pre)   

    y_test=[]
    x_test=[]
    for i in range(length):
        if y_test_pre[i] in y_train_pre:
            y_test.append(y_test_pre[i])
            x_test.append(x_test_pre[i])

    length = len(x_train_pre)   

    y_train=[]
    x_train=[]
    for i in range(length):
        if y_train_pre[i] in y_test:
            y_train.append(y_train_pre[i])
            x_train.append(x_train_pre[i])

    
    sm = SMOTE(k_neighbors=3,random_state=42)

    # x_train, y_train = sm.fit_resample(x_train, y_train)
    # x_train=x_train-np.mean(x_train)
    # scaler = MinMaxScaler()
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)

    dt_clf=decisionTree(x_train,x_test,y_train,y_test)
    rf_clf=random_forest(x_train,x_test,y_train,y_test)
    smo_clf=SMO(x_train,x_test,y_train,y_test)
    ensemble(x_train,x_test,y_train,y_test,dt_clf,rf_clf,smo_clf)


# X_train, X_test, y_train, y_test = train_test_split(urduX, urduY, test_size=0.33, random_state=42)
print("train - Urdu      test-Savee")
classify(urduX,saveeX,urduY,saveeY)
print("train - Urdu      test-emodb")
# classify(urduX,emodbX,urduY,emodbY)
print("train - Urdu      test-emovo")
# classify(urduX,emovoX,urduY,emovoY)


print("train - savee      test-urdu")
classify(saveeX,urduX,saveeY,urduY)
print("train - savee      test-emodb")
# classify(saveeX,emodbX,saveeY,emodbY)
print("train - savee      test-emovo")
# classify(saveeX,emovoX,saveeY,emovoY)

print("train - emodb      test-urdu")
# classify(emodbX,urduX,emodbY,urduY)
print("train - emodb      test-savee")
# classify(emodbX,saveeX,emodbY,saveeY)
print("train - emodb      test-emovo")
# classify(emodbX,emovoX,emodbY,emovoY)

print("train - emovo      test-urdu")
# classify(emovoX,urduX,emovoY,urduY)
print("train - emovo      test-savee")
# classify(emovoX,saveeX,emovoY,saveeY)
print("train - emovo      test-emodb")
# classify(emovoX,emodbX,emovoY,emodbY)
