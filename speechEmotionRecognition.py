import os,glob
import numpy as np 
import librosa
import parselmouth
import tsfel
import torchaudio
import torch
import kaldifeat
from parselmouth.praat import call
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest,chi2
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler



def extractFeatures(fileName):
    y, sr = librosa.load(fileName)

    result=np.array([])

    # extracting chroma features
    stft=np.abs(librosa.stft(y))
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    result=np.hstack((result, chroma))

    # extracting mfcc features
    mfccs=np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    result=np.hstack((result, mfccs))

    # extracting mel features
    mel=np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T,axis=0)
    result=np.hstack((result, mel))

    # extracting contrast features
    S = np.abs(librosa.stft(y))
    contrast = np.mean(librosa.feature.spectral_contrast(S=S, sr=sr).T,axis=0)
    result=np.hstack((result, contrast))

    # extracting zero crossing rate features
    zeroCrossingRate=np.mean(librosa.feature.zero_crossing_rate(y=y).T,axis=0)
    result=np.hstack((result, zeroCrossingRate))

    # extracting tonnetz features
    # hry = librosa.effects.harmonic(y=y)
    # tonnetz = np.mean(librosa.feature.tonnetz(y=hry, sr=sr,chroma=librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0).T,axis=0)
    # result=np.hstack((result, tonnetz))


    # # extracting pitch features
    # pitches, magnitudes = librosa.piptrack(y=y, sr=sr, S=S, fmin=70, fmax=400, n_fft = 4096)
    # pitch = []
    # for i in range(magnitudes.shape[1]):
        # index = magnitudes[:, 1].argmax()
        # pitch.append(pitches[index, i])
    # pitch=np.mean(pitch,axis=0)
    # result=np.hstack((result, pitch))


    # sound = parselmouth.Sound(fileName)

    # # extracting duration features
    # duration = call(sound, 'Get end time')
    # result=np.hstack((result, np.array([duration])))

    # # extracting harmonic features
    # harmonicity = sound.to_harmonicity()
    # harmonicity_values = [call(harmonicity, 'Get value in frame', frame_no)
                              # for frame_no in range(len(harmonicity))]

    # result=np.hstack((result, np.mean(np.array(harmonicity_values),axis=0)))

    # # extracting jitter features
    # pitch = sound.to_pitch()
    # pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    # jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100
    # result=np.hstack((result, np.array([jitter_local])))


    # # extracting shimmer features
    # pitch = sound.to_pitch()
    # pulses = call([sound, pitch], "To PointProcess (cc)")
    # localShimmer =  call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    # result=np.hstack((result, np.array([localShimmer])))


    # extracting energy and entropy
    # F, f_names = ShortTermFeatures.feature_extraction(signal=y, sampling_rate=sr,window=0.050*sr,step= 0.025*sr)
    # result=np.hstack((result, np.mean(F.T,axis=0)))


    # # extracting lpcc features
    # lpccs=tsfel.feature_extraction.features.lpcc(signal=y);
    # result=np.hstack((result, lpccs))

    # # extracting plp features
    # wave, samp_freq = torchaudio.load(fileName)
    # wave = wave.squeeze()
    # wave = wave.flatten()
    # opts = kaldifeat.PlpOptions()
    # opts.mel_opts.num_bins = 23
    # plp = kaldifeat.Plp(opts)
    # plps = plp(wave)
    # result=np.hstack((result, np.mean(np.array(plps),axis=0)))


    print(fileName)
    return result


def loadData(folderName, emotionsConditions):
    x,y = [] , []
    num=150
    for folder in glob.glob(folderName):
        for file in glob.glob(folder+'/*.wav'):
            # num=num-1
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


def decisionTree(x_train,x_test,y_train,y_test):
    
    def train_using_entropy(X_train, X_test, y_train):
      
        clf_entropy = DecisionTreeClassifier(
                criterion = "entropy", random_state = 100,
                max_depth = 3, min_samples_leaf = 5)
      
        clf_entropy.fit(X_train, y_train)
        return clf_entropy
    
    
    def prediction(X_test, clf_object):
      
        y_pred = clf_object.predict(X_test)
        return y_pred
    
    
    
    clf_entropy = train_using_entropy(x_train, x_test, y_train)
    
    print("Results Using Entropy:")
    y_pred_entropy = prediction(x_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


def classify(x_train,x_test_pre,y_train,y_test_pre):

    selector= SelectKBest(chi2, k=130)
    x_train= selector.fit_transform(x_train,y_train)
    x_test_pre=selector.transform(x_test_pre)
    length = len(x_test_pre)   

    y_test=[]
    x_test=[]
    for i in range(length):
        if y_test_pre[i] in y_train:
            y_test.append(y_test_pre[i])
            x_test.append(x_test_pre[i])

    decisionTree(x_train,x_test,y_train,y_test)


# X_train, X_test, y_train, y_test = train_test_split(urduX, urduY, test_size=0.33, random_state=42)
classify(urduX,saveeX,urduY,saveeY)
