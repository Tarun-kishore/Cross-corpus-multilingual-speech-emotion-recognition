import os,glob
import numpy as np 
import pandas as pd
import opensmile
import librosa
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC


#extracting features from audio files
def extractFeatures(fileName):
    y, sr = librosa.load(fileName)

    result=np.array([])

    # extracting chroma features
    stft=np.abs(librosa.stft(y))
    chroma=np.median(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    result=np.hstack((result, chroma))

    # extracting mfcc features
    mfccs=np.median(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)[:20]
    result=np.hstack((result, mfccs))

    # extracting rolloff features
    rolloff = np.median(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99).T,axis=0)
    result=np.hstack((result, rolloff))

    # extracting centroid features
    centroid = np.median(librosa.feature.spectral_centroid(y=y, sr=sr).T,axis=0)
    result=np.hstack((result, centroid))

    # extracting bandwidth features
    bandwidth = np.median(librosa.feature.spectral_bandwidth(y=y, sr=sr).T,axis=0)
    result=np.hstack((result, bandwidth))

    # extracting rms features
    rms = np.median(librosa.feature.rms(y=y).T,axis=0)
    result=np.hstack((result, rms))

    # extracting fundamental frequency
    f0=librosa.pyin(y=y, fmin=librosa.note_to_hz('c2'), fmax=librosa.note_to_hz('c7'))
    f0=np.array(f0)

    f0 = np.median(f0.T,axis=0)
    result=np.hstack((result, f0))


    # extracting contrast features
    S = np.abs(librosa.stft(y))
    contrast = np.median(librosa.feature.spectral_contrast(S=S, sr=sr).T,axis=0)
    result=np.hstack((result, contrast))

    # extracting zero crossing rate features
    zeroCrossingRate=np.median(librosa.feature.zero_crossing_rate(y=y).T,axis=0)
    result=np.hstack((result, zeroCrossingRate))

    # extra eGeMAPS features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    eGeMAPS=smile.process_signal(y,sr)
    eGeMAPS = np.median(eGeMAPS,axis=0)
    result=np.hstack((result, eGeMAPS))

    print(fileName)
    return result


# load all audio files from the dataset folder
def loadData(folderName, emotionsConditions):
    x,y = [] , []
    for folder in glob.glob(folderName):
        for file in glob.glob(folder+'/*.wav'):
            fileName = os.path.basename(file)
            emotion = emotionsConditions(fileName)
            feature = extractFeatures(file)
            x.append(feature)
            y.append(emotion)

    # removing all nan values from features and replacing it
    # with the mean of remaning values of the features
    preprocessor = SimpleImputer(missing_values=np.nan, strategy='mean')
    preprocessor.fit(x)
    x = preprocessor.transform(x)

    # Using SMOTE to oversample data to remove data imbalance
    sm = SMOTE(k_neighbors=3,random_state=42)
    x, y = sm.fit_resample(x, y)

    # Shifting data to have mean value 0
    x=x-np.mean(x)

    # Normalising data : converting data value to value from 0 to 1
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)


    return x,y



# function containing conditions to find emotion using filename in savee dataset
# arguement : filename
# returns : emotion of audio in filename

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

# if the corresponding csv exists then no need ot extract features again, obtain features from the csv file
# if csv file does not exist then extract features and export features and corresponding emotions to csv file

if (os.path.isfile('saveeX.csv')):
    df = pd.read_csv('saveeX.csv', delimiter=',')
    saveeX = [np.array(x) for x in df.values]
    df = pd.read_csv('saveeY.csv')
    saveeY = [x[1] for x in df.values]
else:
    saveeX,saveeY=loadData('data/AudioData/*',saveeConditions)
    df = pd.DataFrame(saveeX)
    df.to_csv('saveeX.csv')
    df = pd.DataFrame(saveeY)
    df.to_csv('saveeY.csv')


# function containing conditions to find emotion using filename in urdu dataset
# arguement : filename
# returns : emotion of audio in filename
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


# if the corresponding csv exists then no need ot extract features again, obtain features from the csv file
# if csv file does not exist then extract features and export features and corresponding emotions to csv file
if (os.path.isfile('urduX.csv')):
    df = pd.read_csv('urduX.csv', delimiter=',')
    urduX = [np.array(x) for x in df.values]
    df = pd.read_csv('urduY.csv', delimiter=',')
    urduY = [x[1] for x in df.values]
else:
    urduX,urduY=loadData('data/Urdu/*',urduConditions)
    df = pd.DataFrame(urduX)
    df.to_csv('urduX.csv')
    df = pd.DataFrame(urduY)
    df.to_csv('urduY.csv')

# function containing conditions to find emotion using filename in emodb dataset
# arguement : filename
# returns : emotion of audio in filename

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

# if the corresponding csv exists then no need ot extract features again, obtain features from the csv file
# if csv file does not exist then extract features and export features and corresponding emotions to csv file
if (os.path.isfile('emodbX.csv')):
    df = pd.read_csv('emodbX.csv', delimiter=',')
    emodbX = [np.array(x) for x in df.values]
    df = pd.read_csv('emodbY.csv', delimiter=',')
    emodbY = [x[1] for x in df.values]
else:
    emodbX,emodbY=loadData('data/emo-db/*',emodbConditions)
    df = pd.DataFrame(emodbX)
    df.to_csv('emodbX.csv')
    df = pd.DataFrame(emodbY)
    df.to_csv('emodbY.csv')

# function containing conditions to find emotion using filename in emovo dataset
# arguement : filename
# returns : emotion of audio in filename

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

# if the corresponding csv exists then no need ot extract features again, obtain features from the csv file
# if csv file does not exist then extract features and export features and corresponding emotions to csv file
if (os.path.isfile('emovoX.csv')):
    df = pd.read_csv('emovoX.csv', delimiter=',')
    emovoX = [np.array(x) for x in df.values]
    df = pd.read_csv('emovoY.csv', delimiter=',')
    emovoY = [x[1] for x in df.values]
else:
    emovoX,emovoY=loadData('data/emovo/EMOVO/*',emovoConditions)
    df = pd.DataFrame(emovoX)
    df.to_csv('emovoX.csv')
    df = pd.DataFrame(emovoY)
    df.to_csv('emovoY.csv')


# calculate accuracy using test results and predicted results of the classifier
# arguements: test dataset results and predicted results 
# returns: void ( prints the accuracy and classification report)

def cal_accuracy(y_test, y_pred):
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))


# SMO classifier
def SMO(x_train, x_test, y_train, y_test):
    svm_model_linear = SVC(gamma='auto')
    clf_return = svm_model_linear
    svm_model_linear.fit(x_train, y_train) 
    svm_predictions = svm_model_linear.predict(x_test) 
    print("Results Using Support Vector Machine: ")
    cal_accuracy(y_test, svm_predictions)
    return clf_return


# decision tree classifier
def decisionTree(x_train,x_test,y_train,y_test):
    
    clf = DecisionTreeClassifier(
                criterion = "entropy", random_state = 100,
                max_depth = 15, min_samples_leaf = 5)
      
    clf_return = clf
    clf.fit(x_train, y_train)
      
    y_pred = clf.predict(x_test)
    
    print("Results Using decision tree:")
    cal_accuracy(y_test, y_pred)

    return clf_return 

# random forest classifier
def random_forest(x_train, x_test, y_train, y_test):

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=100)
    clf_return = model
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Results Using random forest:")
    cal_accuracy(y_test, y_pred)

    return model

# ensemble classifier using majority voting
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


# function takes training and testing data and pass it to different classifiers
def classify(x_train_pre,x_test_pre,y_train_pre,y_test_pre):

    length = len(x_test_pre)   

    # removing unobserved data from training and testing dataset
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

    # feature selection using k best selection method
    selector= SelectKBest(mutual_info_classif, k=35)
    x_train= selector.fit_transform(x_train,y_train)
    x_test=selector.transform(x_test)
    

    dt_clf=decisionTree(x_train,x_test,y_train,y_test)
    rf_clf=random_forest(x_train,x_test,y_train,y_test)
    smo_clf=SMO(x_train,x_test,y_train,y_test)
    ensemble(x_train,x_test,y_train,y_test,dt_clf,rf_clf,smo_clf)  


# Splitting datasets into training and testing dataset
urdu_x_train, urdu_x_test, urdu_y_train, urdu_y_test = train_test_split(urduX, urduY, test_size=0.33, random_state=42)
savee_x_train, savee_x_test, savee_y_train, savee_y_test = train_test_split(saveeX, saveeY, test_size=0.25, random_state=42)
emodb_x_train, emodb_x_test, emodb_y_train, emodb_y_test = train_test_split(emodbX, emodbY, test_size=0.25, random_state=42)
emovo_x_train, emovo_x_test, emovo_y_train, emovo_y_test = train_test_split(emovoX, emovoY, test_size=0.25, random_state=42)


# In corpus tesing : training and testing on same dataset
print("In Corpus Testing")

print("train - Savee      test-Savee")
classify(savee_x_train,savee_x_test,savee_y_train,savee_y_test)

print("train - urdu      test-urdu")
classify(urdu_x_train,urdu_x_test,urdu_y_train,urdu_y_test)

print("train - emodb      test-emodb")
classify(emodb_x_train,emodb_x_test,emodb_y_train,emodb_y_test)

print("train - emovo      test-emovo")
classify(emovo_x_train,emovo_x_test,emovo_y_train,emovo_y_test)


# Cross corpus testing : training and testing on different dataset
print("Cross Corpus Testing")

print("train - Urdu      test-Savee")
classify(urduX,savee_x_test,urduY,savee_y_test)

print("train - Urdu      test-emodb")
classify(urduX,emodb_x_test,urduY,emodb_y_test)

print("train - Urdu      test-emovo")
classify(urduX,emovo_x_test,urduY,emovo_y_test)

print("train - savee      test-urdu")
classify(saveeX,urdu_x_test,saveeY,urdu_y_test)

print("train - emodb      test-urdu")
classify(emodbX,urdu_x_test,emodbY,urdu_y_test)

print("train - emovo      test-urdu")
classify(emovoX,urdu_x_test,emovoY,urdu_y_test)
