# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:25:12 2020

@author: choms
"""
import os
import numpy as np

import cv2
import pyaudio
import wave

import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.io import wavfile
import time
import sys
import seaborn as sns
from scipy import signal

from keras.utils import to_categorical
import pandas as pd

import os.path

from READRECORDSOUND_CLASS import ReadRecordSound_Class
from RECORDSOUND_CLASS import RecoedSound_Class
from SOUNDDEEPLERNIG_CLASS import SoundDeepLerning_Class
from DLLIVEPREDICT_CLASS import DLLivePredict_Class
from CLOUDIOT_CLASS import CloudIOT_Class

import threading

def rec():
    rd=[]   

#    rd.append(RecoedSound_Class("ZZZ","ZERO"))
    rd.append(RecoedSound_Class("ZZZ","NOISE"))
    
#    rd.append(RecoedSound_Class("SORASIT","NAME"))
#    rd.append(RecoedSound_Class("PHATHARASAYA","NAME"))
#    rd.append(RecoedSound_Class("MICHEL","NAME"))
#    rd.append(RecoedSound_Class("DAVID","NAME"))
    
#    rd.append(RecoedSound_Class("TANAPAT","NAME"))
#    rd.append(RecoedSound_Class("KONGSIAM","NAME"))
#    rd.append(RecoedSound_Class("NONGYAW","NAME"))
#    rd.append(RecoedSound_Class("TONY","NAME"))
    
#    rd.append(RecoedSound_Class("XXX","XXX"))
    
    for r in rd :
        for i in range(1000):
            print("Record No. "+ str(i) + " " + str(r.owners)+" , say : "+str(r.sounds))
#            r.Mics(disp=False , recordTime=10 , nperseg=512)#()#(disp=True)
#            r.Mics_section(recordTime=60 , sammeword=True )
            r.Mics_whole(recordTime=8)
#            r.Mics(disp=True , recordTime=8 , nperseg=512 , ZERO=False)#()#(disp=True)
#            r.Mics_whole1(disp=True,recordTime=10)
#            time.sleep(1)
            
#            r.Mics(disp=False , recordTime=10 , nperseg=1024*2 , ZERO=True)#()#(disp=True)
#            r.Mics(disp=False , recordTime=10 , nperseg=1024*2 , ZERO=False)#()#(disp=True)
#            r.Mics_whole1(disp=True,recordTime=10)
            time.sleep(0.5)
            
def rec1():
    rd=[]   

#    rd.append(RecoedSound_Class("ZZZ","ZERO"))
#    rd.append(RecoedSound_Class("ZZZ","NOISE"))
    rd.append(RecoedSound_Class("SAMPLE9","SAMPLE9"))

    
#    rd.append(RecoedSound_Class("XXX","XXX"))
    
    for r in rd : 
        for i in range(1004):
            print("Record No. "+ str(i) + " " + str(r.owners)+" , say : "+str(r.sounds))
            
            r.Mics_whole1(disp=True,recordTime=10, noise=500)
#            r.Mics_whole1Noise(disp=True, recordTime=10,noise=5000)
            time.sleep(0.5)   


def rec_training_model2():    
    
    snd_json = []
    snd_x_train = []
    snd_y_train = []
    snd_names = ""
    
    WAV=ReadRecordSound_Class()    
#    WAV.readRecordWAV(disp=False , output=False , lowpass=6000 , nperseg=512)
    WAV.readRecordWAV1(disp=False , output=False , lowpass=6000 , nperseg=512)
    #yyyy = np.array(WAV.Y_DATASET) #un used
    
#    print(len(WAV.X_DATASET))
#    print(len(WAV.X_DATASET[0]))
##    print((WAV.X_DATASET[0]))
#    print(len(WAV.X_DATASET[0][0]))
##    print((WAV.X_DATASET[0][0]))
#    print(len(WAV.X_DATASET[0][0][0]))
##    print((WAV.X_DATASET[0][0][0]))
#    print(len(WAV.X_DATASET[0][0][0][0]))
##    print((WAV.X_DATASET[0][0][0][0]))
    
    xxxx = np.array(WAV.X_DATASET)
    
    print("****") 
    print(xxxx.shape) 
    idx_owners=0
    idx_sounds=0
    
#    for owners in range(len(WAV.X_DATASET)):
#        pass
#        print(WAV.OWNERS[idx_owners])
#        for sounds in range(len(WAV.X_DATASET[0])):
#            pass
#            print(idx_sounds)  #used
#            print(WAV.SOUNDS[idx_sounds])
#            for files in range(len(WAV.X_DATASET[0][0])):
#                pass
#                l=np.array(WAV.X_DATASET[owners][sounds][files])
#                print(l.shape)
#                
#            idx_sounds=idx_sounds+1
#                
#        idx_owners=idx_owners+1   
#    
#    sys.exit()
    
    for owners in xxxx:
        print(idx_owners)
        print(WAV.OWNERS[idx_owners])
        
        for sounds in owners:
            print(idx_sounds)  #used
            print(WAV.SOUNDS[idx_sounds])
            #train deep lerning here
            snd_x_train.extend(sounds)
            snd_y_train.extend(to_categorical(np.ones((len(sounds),), dtype=int)*idx_sounds, num_classes = len(WAV.SOUNDS)))
            snd_names=str(snd_names) + str(WAV.OWNERS[idx_owners]) + str(WAV.SOUNDS[idx_sounds])
            snd_json.append([ idx_sounds , str(WAV.OWNERS[idx_owners]) , str(WAV.SOUNDS[idx_sounds]) , str(WAV.FRAMESIZE)  ])
            
            for files in sounds:
                print(files.shape)

            idx_sounds=idx_sounds+1
                
        idx_owners=idx_owners+1   
       
    snd_names = WAV.saveModelNameWithData(snd_names , "classify",snd_json)
#    sys.exit()
    
    print("**TRAIN MODEL**") 
    print(snd_names)
#    print(snd_json)
    
#    sys.exit()
    
    y=np.array(snd_y_train).astype('int')
    x=np.array(snd_x_train).astype('float')
    print(x.shape)
    print(y.shape)

#    sys.exit()

    snd_model = SoundDeepLerning_Class()
    snd_model.X_train=x
    snd_model.Y_train=y
    snd_model.ModelName_train=snd_names
    snd_model.TrainModel()

    snd1=SoundDeepLerning_Class()
    print(snd_names)
#    snd1.TestModel(snd_names,np.array(snd_x_train[0:10]).astype('float'),np.array(snd_y_train[0:10]).astype('int'))
    snd1.TestModel(snd_names,np.array(snd_x_train).astype('float'),np.array(snd_y_train).astype('int'))


def rec_training_model3():    
    
    snd_json = []
    snd_x_train = []
    snd_y_train = []
    snd_names = ""
    
    WAV=ReadRecordSound_Class()    
#    WAV.readRecordWAV(disp=False , output=False , lowpass=6000 , nperseg=512)
#    WAV.readRecordWAV1(disp=False , output=False , lowpass=6000 , nperseg=512)
    
    
    logScl=False
    extensionName = "_yesLogScl_"
    if not(logScl) : extensionName = "_noLogScl_" 
#    WAV.readRecordWAV1compress(disp=True , output=False , lowpass=0 , nperseg=1024*10 , compressed=100, logScl=logScl)
    WAV.readRecordWAV1compress(disp=False , output=False , lowpass=0 , nperseg=512*200 , compressed=200, logScl=logScl)
    #yyyy = np.array(WAV.Y_DATASET) #un used

    
#    print(len(WAV.X_DATASET))
#    print(len(WAV.X_DATASET[0]))
##    print((WAV.X_DATASET[0]))
#    print(len(WAV.X_DATASET[0][0]))
##    print((WAV.X_DATASET[0][0]))
#    print(len(WAV.X_DATASET[0][0][0]))
##    print((WAV.X_DATASET[0][0][0]))
#    print(len(WAV.X_DATASET[0][0][0][0]))
##    print((WAV.X_DATASET[0][0][0][0]))
    
    xxxx = np.array(WAV.X_DATASET)
    
    print("****") 
    print(xxxx.shape) 
    idx_owners=0
    idx_sounds=0
    
#    sys.exit()
    
#    for owners in range(len(WAV.X_DATASET)):
#        pass
#        print(WAV.OWNERS[idx_owners])
#        for sounds in range(len(WAV.X_DATASET[0])):
#            pass
#            print(idx_sounds)  #used
#            print(WAV.SOUNDS[idx_sounds])
#            for files in range(len(WAV.X_DATASET[0][0])):
#                pass
#                l=np.array(WAV.X_DATASET[owners][sounds][files])
#                print(l.shape)
#                
#            idx_sounds=idx_sounds+1
#                
#        idx_owners=idx_owners+1   
#    
#    sys.exit()
    
    for owners in xxxx:
        print(idx_owners)
        print(WAV.OWNERS[idx_owners])
        
        for sounds in owners:
            print(idx_sounds)  #used
            print(WAV.SOUNDS[idx_sounds])
            #train deep lerning here
            snd_x_train.extend(sounds)
            snd_y_train.extend(to_categorical(np.ones((len(sounds),), dtype=int)*idx_sounds, num_classes = len(WAV.SOUNDS)))
            snd_names=str(snd_names) + str(WAV.OWNERS[idx_owners]) + str(WAV.SOUNDS[idx_sounds])
            snd_json.append([ idx_sounds , str(WAV.OWNERS[idx_owners]) , str(WAV.SOUNDS[idx_sounds]) , str(WAV.FRAMESIZE)  ])
            
            for files in sounds:
                print(files.shape)

            idx_sounds=idx_sounds+1
                
        idx_owners=idx_owners+1   
       
    snd_names = WAV.saveModelNameWithData(extensionName+snd_names , "classify_00_new" , snd_json) #logScl=True

#    sys.exit()
    
    print("**TRAIN MODEL**") 
    print(snd_names)
#    print(snd_json)
    
#    sys.exit()
    
    y=np.array(snd_y_train).astype('int')
    x=np.array(snd_x_train).astype('float')
    print(x.shape)
    print(y.shape)

#    sys.exit()

    snd_model = SoundDeepLerning_Class()
    snd_model.X_train=x
    snd_model.Y_train=y
    snd_model.ModelName_train=snd_names
    snd_model.TrainModel()

    snd1=SoundDeepLerning_Class()
    print(snd_names)
#    snd1.TestModel(snd_names,np.array(snd_x_train[0:10]).astype('float'),np.array(snd_y_train[0:10]).astype('int'))
    snd1.TestModel(snd_names,np.array(snd_x_train).astype('float'),np.array(snd_y_train).astype('int'))



    
    
def rec_load_model1():    
    
    print("**LOAD MODEL**") 
    ld = []
    ld_labal = []
    ld_owners = []
    ld_sounds = []
    ld_models = []
    ld_framesz = []
 
    WAV=ReadRecordSound_Class()    
    WAV.readModelWithData()
    i=0
    for labal,owners,sounds,models,framesz in zip(WAV.MODELLABEL,WAV.MODELOWNERS,WAV.MODELSOUNDS,WAV.MODELS,WAV.FRAMESZ) :
        print(labal)
        print(owners)
        print(sounds)
        print(models)
        print(framesz)
        ld_labal.append(labal)
        ld_owners.append(owners)
        ld_sounds.append(sounds)
        ld_models.append(models) 
        ld_framesz.append(framesz) 
        
        ld.append(SoundDeepLerning_Class())
        ld[i].Label_load=labal
        ld[i].Owners_load=owners
        ld[i].Sounds_load=sounds
        ld[i].ModelName_load=models
        ld[i].LoadModel()
        
        i=i+1
        
    
    testMics = RecoedSound_Class("test","test")
#    sptg=testMics.TestMics( disp=False , save=False ,recordTime=10 , lowpass=6000 , nperseg=512) 
#    sptg=testMics.WAV(disp=True , output=True , lowpass=6000 , name=".\\SNDPATH\\DAVID\\NAME\\10.wav" , nperseg=512 )
#    sptg=testMics.WAV1(disp=True , output=True , lowpass=6000 , name=".\\SNDPATH\\DAVID\\NAME\\10.wav" , nperseg=512 , size=int(framesz[0])  )

    frames_data,frame_size,dataPerFrame,FORMAT,CHANNELS,RATE=testMics.signelWAV1(name=".\\SNDPATH\\SORASIT\\NAME\\10.wav" )
#    frames_data,frame_size,dataPerFrame,FORMAT,CHANNELS,RATE=testMics.signelTestMics_whole( recordTime=8 )

#    sizey=sptg.shape[0]
#    sizex=sptg.shape[1]
#    b=sptg.reshape((1, sizey , sizex)).astype('float')
#    print(b.shape)
    
#    b=[]
#    b.append(sptg)
#    bb=np.array(b)
#    print(bb.shape)
    
    final_label = -1
    final_plabel = 0.7
    final_owner = ""
    final_sound = ""
    
    tmp = frames_data
    
    for LD,labal,owners,sounds,models,framesz in zip(ld,ld_labal,ld_owners,ld_sounds,ld_models,ld_framesz) :
#        LD.Label_load=labal
#        LD.Owners_load=owners
#        LD.Sounds_load=sounds
#        LD.ModelName_load=models
#        LD.LoadModel()
        
        b=[]
        if frame_size==0 : continue
#        print("frames_data : " + str(len(frames_data)))
#        print("temp : " + str(len(tmp)))
#        print("frame_size : " + str(frame_size))
        b.append(testMics.spectrogramWAV1(disp=True 
                                          , output=True 
                                          , lowpass=6000 
                                          , nperseg=512 
                                          , size=framesz[0] 
                                          , frames_data=tmp 
                                          , frame_size=frame_size
                                          , dataPerFrame=dataPerFrame
                                          , FORMAT=FORMAT 
                                          , CHANNELS=CHANNELS
                                          , RATE=RATE))
        bb=np.array(b)
        print(bb.shape)
        
        lab,plab=LD.TestLoadModel(bb,0.5)
        print(LD.Owners_load[int(lab)])
        print(plab)
        
        
        if plab > final_plabel : 
            final_label=lab
            final_plabel=plab
            final_owner=LD.Owners_load[int(lab)]
            final_sound=LD.Sounds_load[int(lab)]
         
    if final_label > -1 :    
        print(final_label)
        print(final_plabel)
        print(final_owner)
        print(final_sound)
    else:
        print("Don't know !")

    print("****") 


def rec_load_model2():    
    
    print("**LOAD MODEL**") 
    ld = []
    ld_labal = []
    ld_owners = []
    ld_sounds = []
    ld_models = []
    ld_framesz = []
 
    WAV=ReadRecordSound_Class()    
    WAV.readModelWithData()
    i=0
    for labal,owners,sounds,models,framesz in zip(WAV.MODELLABEL,WAV.MODELOWNERS,WAV.MODELSOUNDS,WAV.MODELS,WAV.FRAMESZ) :
        print(labal)
        print(owners)
        print(sounds)
        print(models)
        print(framesz)
        ld_labal.append(labal)
        ld_owners.append(owners)
        ld_sounds.append(sounds)
        ld_models.append(models) 
        ld_framesz.append(framesz) 
        
        ld.append(SoundDeepLerning_Class())
        ld[i].Label_load=labal
        ld[i].Owners_load=owners
        ld[i].Sounds_load=sounds
        ld[i].ModelName_load=models
        ld[i].LoadModel()
        
        i=i+1
        
    
    testMics = RecoedSound_Class("test","test")
#    sptg=testMics.TestMics( disp=False , save=False ,recordTime=10 , lowpass=6000 , nperseg=512) 
#    sptg=testMics.WAV(disp=True , output=True , lowpass=6000 , name=".\\SNDPATH\\DAVID\\NAME\\10.wav" , nperseg=512 )
#    sptg=testMics.WAV1(disp=True , output=True , lowpass=6000 , name=".\\SNDPATH\\DAVID\\NAME\\10.wav" , nperseg=512 , size=int(framesz[0])  )

    frames_data,frame_size,dataPerFrame,FORMAT,CHANNELS,RATE=testMics.signelWAV1(name=".\\SNDPATH\\SORASIT\\NAME\\10.wav" )
#    frames_data,frame_size,dataPerFrame,FORMAT,CHANNELS,RATE=testMics.signelTestMics_whole( recordTime=8 )

#    sizey=sptg.shape[0]
#    sizex=sptg.shape[1]
#    b=sptg.reshape((1, sizey , sizex)).astype('float')
#    print(b.shape)
    
#    b=[]
#    b.append(sptg)
#    bb=np.array(b)
#    print(bb.shape)
    
    final_label = -1
    final_plabel = 0.7
    final_owner = ""
    final_sound = ""
    
    tmp = frames_data
    
    for LD,labal,owners,sounds,models,framesz in zip(ld,ld_labal,ld_owners,ld_sounds,ld_models,ld_framesz) :
#        LD.Label_load=labal
#        LD.Owners_load=owners
#        LD.Sounds_load=sounds
#        LD.ModelName_load=models
#        LD.LoadModel()
        
        logScl=True
        if("_noLogScl" in models):
            logScl=False
#                continue;
        
        b=[]
        if frame_size==0 : continue
#        print("frames_data : " + str(len(frames_data)))
#        print("temp : " + str(len(tmp)))
#        print("frame_size : " + str(frame_size))
        b.append(testMics.spectrogramWAV1compress(
                                            disp=True 
                                          , output=True 
                                          , lowpass=0 
                                          , nperseg=1024*2
                                          , size=framesz[0] 
                                          , frames_data=tmp 
                                          , frame_size=frame_size
                                          , dataPerFrame=dataPerFrame
                                          , FORMAT=FORMAT 
                                          , CHANNELS=CHANNELS
                                          , RATE=RATE
                                          , compressed=10
                                          , logScl=logScl
                                          ))
        bb=np.array(b)
        print(bb.shape)
        
        lab,plab=LD.TestLoadModel(bb,0.5)
        print(LD.Owners_load[int(lab)])
        print(plab)
        
        
        if plab > final_plabel : 
            final_label=lab
            final_plabel=plab
            final_owner=LD.Owners_load[int(lab)]
            final_sound=LD.Sounds_load[int(lab)]
         
    if final_label > -1 :    
        print(final_label)
        print(final_plabel)
        print(final_owner)
        print(final_sound)
    else:
        print("Don't know !")

    print("****") 




def live_load_model():
    Test=DLLivePredict_Class()
    Test.loadDLmodel()
    Test.signelLiveMics_whole(recordTime=0)
#    Test.signelLiveMics_whole1(recordTime=10)
    
 
def live_load_model1():    
    ##global share_flag
    ct = CloudIOT_Class() 
    ct.googleSheet_init()
    ct.googleSheet_responseServerThread()
    print("exit")
    #print(share_flag)
    
    Test=DLLivePredict_Class()
    Test.loadDLmodel()
    #Test.signelLiveMics_whole(recordTime=0)
    #Test.signelLiveMics_whole1(recordTime=10)
    
    thr=threading.Thread(target=Test.signelLiveMics_whole1, args=[10 , 500])
    thr.start()


##### Main ########
    
#rec1()   #For record the sounds
#rec_training_model3()   #For make the deep learnin model from the record's sounds
live_load_model1()   #For test the predict DL Model from mics on live.