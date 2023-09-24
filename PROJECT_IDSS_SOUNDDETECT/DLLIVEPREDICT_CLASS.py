# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:37:58 2020

@author: choms
"""

import os
import numpy as np

import cv2
import pyaudio
import wave
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
import os
import matplotlib
matplotlib.use('Agg')

from sys import byteorder
from array import array
from struct import pack

import threading
from collections import Counter


from READRECORDSOUND_CLASS import ReadRecordSound_Class
from RECORDSOUND_CLASS import RecoedSound_Class
from SOUNDDEEPLERNIG_CLASS import SoundDeepLerning_Class
from CLOUDIOT_CLASS import CloudIOT_Class


class DLLivePredict_Class():
    
    
    def __init__(self
                 ):
        pass
        self.ld = []
        self.ld_labal = []
        self.ld_owners = []
        self.ld_sounds = []
        self.ld_models = []
        self.ld_framesz = []
        
        self.cldiot = CloudIOT_Class()  
        self.cldiot.googleSheet_init()
        self.testMics1 = RecoedSound_Class("test","test")
        
        self.acc_result = [-1,-1,-1]
        self.acc_count = 0
        
        
    
    def signelLiveMics_whole(self, recordTime=0 ):  
        pass
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = int(recordTime/2)  #5s record time
          
        p = pyaudio.PyAudio()
#        print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        frames0 = []
        
#        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        frameNumber=int((RATE / CHUNK) * RECORD_SECONDS)
        frameNoise=int((RATE / CHUNK) * 3) #6s
        frameThreadWait=int((RATE / CHUNK) * 1) #1s
        f=0
        snd_start=False
        Noise=500
        
        while True:
            data = stream.read(CHUNK)
            data_chunk=array('h',data)
            vol=max(data_chunk)
            if(vol>=Noise):
#                print("something said " + str(f) +" " + str(vol))
                Noise=vol
            if(f>=frameNoise): break
            f=f+1
            
        print("* start Mics on Live")
        f=0
        fd=0
        threadflag = False
        fr_cnt=0
        while True:
            data = stream.read(CHUNK)
            data_chunk=array('h',data)
            vol=max(data_chunk)
            if(vol>=Noise):
                print("something said " + str(f))
                frames.append(data)
                snd_start=True
                fd=f
                fr_cnt=fr_cnt+1
            else:
                pass
                if snd_start : 
                    frames0.append(frames)
                    frames = []
                    snd_start=False
                    threadflag=True
                
                if threadflag :
                    if((f-fd)>frameThreadWait) :
                        if ( fr_cnt>10 ) :
                            print("call thread here")
                            print(str(frameThreadWait)+" : "+str(fr_cnt)+" : "+str(f-fd))
                            print()
    #                        thr=threading.Thread(target=self.callDLThread, args=[frames0,FORMAT,CHANNELS,RATE])
    #                        thr.start()
                            self.callDLThread(frames=frames0,FORMAT=FORMAT,CHANNELS=CHANNELS,RATE=RATE)
                        frames0 = []
                        fr_cnt=0
                        threadflag=False
                        
#                print("nothing")
#            print("\n")
            if frameNumber != 0 :
                if(f>=frameNumber): break
            f=f+1
        
        print("* stop Mics on Live")
        
        #        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()


    def callDLThread(self,frames=[] , FORMAT=pyaudio.paInt16, CHANNELS=2 ,RATE=44100 ):
        
        frames0=[]
        frames0.extend(frames)
        
        if(len(frames0)==0) : 
            print("No Sound detected !")
            return [],0,0,FORMAT,CHANNELS,RATE
        

        
        frames00 = []       
        for frame_section in frames0 :
            pass
            frames00.extend(frame_section)
        
        
        
        ss=0
        frames1=[]
        for amp in frames00 :
            pass
            data1 = np.fromstring(amp,dtype=np.int16)
            frames1.extend(data1)
            ss=ss+1
            
            
        frames_data=frames1
        frame_size=ss
        dataPerFrame=len(data1)
        
#        self.testDLmodel(framesdata=frames_data , frame_size=frame_size , dataPerFrame=dataPerFrame , FORMAT=FORMAT, CHANNELS=CHANNELS ,RATE=RATE )
#        thr=threading.Thread(target=self.testDLmodel, args=[frames_data , frame_size , dataPerFrame , FORMAT, CHANNELS ,RATE])
#        thr.start()
#        self.testDLmodel1(framesdata=frames_data , frame_size=frame_size , dataPerFrame=dataPerFrame , FORMAT=FORMAT, CHANNELS=CHANNELS ,RATE=RATE )
#        self.testDLmodel1compress(framesdata=frames_data , frame_size=frame_size , dataPerFrame=dataPerFrame , FORMAT=FORMAT, CHANNELS=CHANNELS ,RATE=RATE )
        self.testDLmodel2compress(framesdata=frames_data , frame_size=frame_size , dataPerFrame=dataPerFrame , FORMAT=FORMAT, CHANNELS=CHANNELS ,RATE=RATE )

    def loadDLmodel(self):    
        
        print("**LOAD MODEL**") 
     
        WAV=ReadRecordSound_Class()    
        WAV.readModelWithData()
        i=0
        for labal,owners,sounds,models,framesz in zip(WAV.MODELLABEL,WAV.MODELOWNERS,WAV.MODELSOUNDS,WAV.MODELS,WAV.FRAMESZ) :
            print(labal)
            print(owners)
            print(sounds)
            print(models)
            print(framesz)
            self.ld_labal.append(labal)
            self.ld_owners.append(owners)
            self.ld_sounds.append(sounds)
            self.ld_models.append(models) 
            self.ld_framesz.append(framesz) 
            
            self.ld.append(SoundDeepLerning_Class())
            self.ld[i].Label_load=labal
            self.ld[i].Owners_load=owners
            self.ld[i].Sounds_load=sounds
            self.ld[i].ModelName_load=models
            self.ld[i].LoadModel()
            
            i=i+1
            
        
#        testMics = RecoedSound_Class("test","test")
#        frames_data,frame_size,dataPerFrame,FORMAT,CHANNELS,RATE=testMics.signelWAV1(name=".\\SNDPATH\\TANAPAT\\NAME\\10.wav" )
#        frames_data,frame_size,dataPerFrame,FORMAT,CHANNELS,RATE=testMics.signelTestMics_whole( recordTime=8 )
    
    
    
    def testDLmodel(self,framesdata=[] , frame_size=0 , dataPerFrame=0 , FORMAT=pyaudio.paInt16, CHANNELS=2 ,RATE=44100 ):    
  
        
        frames_data=[]
        frames_data.extend(framesdata)
        
        testMics = RecoedSound_Class("test","test")
        
        final_label = -1
        final_plabel = 0.7
        final_owner = ""
        final_sound = ""
        
        for LD,labal,owners,sounds,models,framesz in zip(self.ld,self.ld_labal,self.ld_owners,self.ld_sounds,self.ld_models,self.ld_framesz) :
    #        LD.Label_load=labal
    #        LD.Owners_load=owners
    #        LD.Sounds_load=sounds
    #        LD.ModelName_load=models
    #        LD.LoadModel()
            
            b=[]
            if frame_size==0 : continue
            b.append(testMics.spectrogramWAV1(disp=False 
                                              , output=True 
                                              , lowpass=6000 
                                              , nperseg=512 
                                              , size=framesz[0] 
                                              , frames_data=frames_data 
                                              , frame_size=frame_size
                                              , dataPerFrame=dataPerFrame
                                              , FORMAT=FORMAT 
                                              , CHANNELS=CHANNELS
                                              , RATE=RATE))
            bb=np.array(b)
            print(bb.shape)
            
            lab,plab=LD.TestLoadModel(bb,0.5) # cant run on thread
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
    
        
    
    def testDLmodel1(self,framesdata=[] , frame_size=0 , dataPerFrame=0 , FORMAT=pyaudio.paInt16, CHANNELS=2 ,RATE=44100 ):    
  
        flag=""
        frames_data=[]
        frames_data.extend(framesdata)
        
#        self.testMics1 = RecoedSound_Class("test","test")
        
        final_label = -1
        final_plabel = 0.998
        final_owner = ""
        final_sound = ""
        
        for LD,labal,owners,sounds,models,framesz in zip(self.ld,self.ld_labal,self.ld_owners,self.ld_sounds,self.ld_models,self.ld_framesz) :
    #        LD.Label_load=labal
    #        LD.Owners_load=owners
    #        LD.Sounds_load=sounds
    #        LD.ModelName_load=models
    #        LD.LoadModel()
            
            b=[]
            if frame_size==0 : continue
            b.append(self.testMics1.spectrogramWAV2(disp=True 
                                              , output=True 
                                              , lowpass=6000 
                                              , nperseg=512 
                                              , size=framesz[0] 
                                              , frames_data=frames_data 
                                              , frame_size=frame_size
                                              , dataPerFrame=dataPerFrame
                                              , FORMAT=FORMAT 
                                              , CHANNELS=CHANNELS
                                              , RATE=RATE))
            
            i=self.cldiot.share_int_get()
            if i==4 :
                flag=self.cldiot.share_flag_get()
                print(flag)
            
            if i==5 :
                flag=self.cldiot.share_flag_get()
                print(flag)
                return
            
            bb=np.array(b)
            print(bb.shape)
            
            lab,plab=LD.TestLoadModel(bb,0.5) # cant run on thread
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
            thr=threading.Thread(target=self.cloudIotThred, args=[final_label,final_plabel,final_owner,final_sound])
            thr.start()
            self.testMics1.displayResult(
                      disp=True ,
                      pics="IDSS.png",
                      final_label=final_label ,
                      final_plabel=final_plabel ,
                      final_owner=final_owner ,
                      final_sound=final_sound ,
                      col=(0,255,0)
                      )
            
        else:
            print("Don't know !")
            self.testMics1.displayResult(
                      disp=True ,
                      pics="IDSS.png",
                      final_label=final_label ,
                      final_plabel=final_plabel ,
                      final_owner="Not able to pedict the result" ,
                      final_sound="Probability is not acceptable" 
                      )
    
        print("****") 
        
        
    def cloudIotThred(self
                      ,final_label=0
                      , final_plabel=0
                      , final_owner=""
                      , final_sound="" 
                      ):   
        global final_owner_flag
#        message="IDSS monitor"
        message="Monitor"
        if final_owner=="ZZZ" :
            message=""

        if final_owner=="SORASIT" :
            message="MachineNumber1 : Warning on arm side arm but still continue running. Pls change the roller of arm"
            
        if message != "" :
#            self.cldiot = CloudIOT_Class()  
#            self.cldiot.googleSheet_init()
            self.cldiot.googleSheet_updateLog1(final_owner,final_sound,message)
           
            if final_owner=="SAMPLE9" : 
                # self.cldiot.googleMail(subject='pls attention : MachineNumber1')
                self.cldiot.LINEnotify(subject='pls attention : MachineNumber1')
                pass
            final_owner_flag=final_owner
            
            
            
    
    def signelLiveMics_whole1(self, recordTime=10 , noise=500 ):  

        for j in range(3) :
            self.testMics1.displayResult1(
                      disp=True ,
                      pics="IDSS.png",
                      final_label="" ,
                      final_plabel="" ,
                      final_owner="" ,
                      final_sound="" ,
                      message=""
                      )
        offflag=True

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = int(recordTime/2)  #5s record time
          
        p = pyaudio.PyAudio()
#        print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        frames0 = []
        
#        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        frameNumber=int((RATE / CHUNK) * RECORD_SECONDS)
        frameNoise=int((RATE / CHUNK) * 3) #6s
        frameThreadWait=int((RATE / CHUNK) * 1) #1s
        f=0
        snd_start=False
        Noise=500
        
        if noise==0 :
            while True:
                data = stream.read(CHUNK)
                data_chunk=array('h',data)
                vol=max(data_chunk)
                if(vol>=Noise):
    #                print("something said " + str(f) +" " + str(vol))
                    Noise=vol
                if(f>=frameNoise): break
                f=f+1
        else :
            Noise=noise
            
        print("* start Mics on Live")
        f=0
        fd=0
        threadflag = False
        fr_cnt=0
        while True:
            
            i=self.cldiot.share_int_get()
            if i==1 :
                flag=self.cldiot.share_flag_get()
                print(flag)
                break
            
            if i==2 :
#                flag=self.cldiot.share_flag_get()
#                print(flag)
                offflag=True

                
            if i==3 :
#                flag=self.cldiot.share_flag_get()
#                print(flag)
                if offflag :
                    for j in range(3) :
                        self.testMics1.displayResult1(
                                  disp=True ,
                                  pics="IDSS.png",
                                  final_label="" ,
                                  final_plabel="" ,
                                  final_owner="" ,
                                  final_sound="" ,
                                  message=""
                                  )
                    offflag=False
                continue
            
            if i==4 :
#                flag=self.cldiot.share_flag_get()
#                print(flag)
                 offflag=True
 
            
            if i==5 :
#                flag=self.cldiot.share_flag_get()
#                print(flag)
                if offflag :
                    for j in range(3) :
                        self.testMics1.displayResult1(
                                  disp=True ,
                                  pics="IDSS.png",
                                  final_label="" ,
                                  final_plabel="" ,
                                  final_owner="" ,
                                  final_sound="" ,
                                  message=""
                                  )
                    offflag=False

            
            data = stream.read(CHUNK)
            data_chunk=array('h',data)
            vol=max(data_chunk)
            if(vol>=Noise) and ( fr_cnt<frameNumber ):
                print("Mics detect " + str(f))
                frames.append(data)
                snd_start=True
                fd=f
                fr_cnt=fr_cnt+1
            else:
                pass
                if snd_start : 
                    frames0.append(frames)
                    frames = []
                    snd_start=False
                    threadflag=True
                
                if threadflag :
                    if((f-fd)>frameThreadWait) :
                        if ( fr_cnt>=frameNumber ) :
                            print("call thread here")
                            print(str(frameThreadWait)+" : "+str(fr_cnt)+" : "+str(f-fd))
                            print()
    #                        thr=threading.Thread(target=self.callDLThread, args=[frames0,FORMAT,CHANNELS,RATE])
    #                        thr.start()
                            self.callDLThread(frames=frames0,FORMAT=FORMAT,CHANNELS=CHANNELS,RATE=RATE)
                        frames0 = []
                        fr_cnt=0
                        threadflag=False
                        
#                print("nothing")
#            print("\n")
#            if frameNumber != 0 :
#                if(f>=frameNumber): break
            f=f+1
        
        print("* stop Mics on Live")
        print("****signelLiveMics THREAD EXIT****")
        
        #        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        self.testMics1.destroydisplay()
            
        
    def testDLmodel1compress(self,framesdata=[] , frame_size=0 , dataPerFrame=0 , FORMAT=pyaudio.paInt16, CHANNELS=2 ,RATE=44100 ):    
  
        flag=""
        frames_data=[]
        frames_data.extend(framesdata)
        
#        self.testMics1 = RecoedSound_Class("test","test")
        
        final_label = -1
        final_plabel = 0.95
        final_owner = ""
        final_sound = ""
        
        for LD,labal,owners,sounds,models,framesz in zip(self.ld,self.ld_labal,self.ld_owners,self.ld_sounds,self.ld_models,self.ld_framesz) :
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
            b.append(self.testMics1.spectrogramWAV2compress(
                                                disp=True 
                                              , output=False 
                                              , lowpass=0 
                                              , nperseg=512*200
                                              , size=framesz[0] 
                                              , frames_data=frames_data 
                                              , frame_size=frame_size
                                              , dataPerFrame=dataPerFrame
                                              , FORMAT=FORMAT 
                                              , CHANNELS=CHANNELS
                                              , RATE=RATE
                                              , compressed=200
                                              , logScl=logScl
                                              ))
            
            i=self.cldiot.share_int_get()
            if i==4 :
                flag=self.cldiot.share_flag_get()
                print(flag)
            
            if i==5 :
                flag=self.cldiot.share_flag_get()
                print(flag)
                return
            
            bb=np.array(b)
            print(bb.shape)
            
            lab,plab=LD.TestLoadModel(bb,0.5) # cant run on thread
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
            thr=threading.Thread(target=self.cloudIotThred, args=[final_label,final_plabel,final_owner,final_sound])
            thr.start()
            self.testMics1.displayResult(
                      disp=True ,
                      pics="IDSS.png",
                      final_label=final_label ,
                      final_plabel=final_plabel ,
                      final_owner=final_owner ,
                      final_sound=final_sound ,
                      col=(0,255,0)
                      )
            
        else:
            print("Don't know !")
            self.testMics1.displayResult(
                      disp=True ,
                      pics="IDSS.png",
                      final_label=final_label ,
                      final_plabel=final_plabel ,
                      final_owner="Not able to pedict the result" ,
                      final_sound="Probability is not acceptable" 
                      )
    
        print("****") 
        
        
        
    def testDLmodel2compress(self,framesdata=[] , frame_size=0 , dataPerFrame=0 , FORMAT=pyaudio.paInt16, CHANNELS=2 ,RATE=44100 ):    
  
        flag=""
        frames_data=[]
        frames_data.extend(framesdata)
        
#        self.testMics1 = RecoedSound_Class("test","test")
        
        final_label = -1
        final_plabel = 0.8
        final_owner = ""
        final_sound = ""
        just4show = -1
        
        final_label_lt = []
        final_plabel_lt = []
        final_owner_lt = []
        final_sound_lt = []
        
        firstFlag=True
        for LD,labal,owners,sounds,models,framesz in zip(self.ld,self.ld_labal,self.ld_owners,self.ld_sounds,self.ld_models,self.ld_framesz) :
    #        LD.Label_load=labal
    #        LD.Owners_load=owners
    #        LD.Sounds_load=sounds
    #        LD.ModelName_load=models
    #        LD.LoadModel()
            
    #                continue;

            b=[]
            if frame_size==0 : continue
            if firstFlag :
                noLogScl=self.testMics1.spectrogramWAV2compress(
                                                    disp=True 
                                                  , output=False 
                                                  , lowpass=0 
                                                  , nperseg=512*200
                                                  , size=framesz[0] 
                                                  , frames_data=frames_data 
                                                  , frame_size=frame_size
                                                  , dataPerFrame=dataPerFrame
                                                  , FORMAT=FORMAT 
                                                  , CHANNELS=CHANNELS
                                                  , RATE=RATE
                                                  , compressed=200
                                                  , logScl=False
                                                  )
                yesLogScl=self.testMics1.spectrogramWAV2compress(
                                                    disp=True 
                                                  , output=False 
                                                  , lowpass=0 
                                                  , nperseg=512*200
                                                  , size=framesz[0] 
                                                  , frames_data=frames_data 
                                                  , frame_size=frame_size
                                                  , dataPerFrame=dataPerFrame
                                                  , FORMAT=FORMAT 
                                                  , CHANNELS=CHANNELS
                                                  , RATE=RATE
                                                  , compressed=200
                                                  , logScl=True
                                                  )
                firstFlag=False
              
            if("noLogScl" in models):
                b.append(noLogScl)
            else:
                b.append(yesLogScl)


            
            i=self.cldiot.share_int_get()
            if i==4 :
                flag=self.cldiot.share_flag_get()
                print(flag)
            
            if i==5 :
                flag=self.cldiot.share_flag_get()
                print(flag)
                return
            
            bb=np.array(b)
            print(bb.shape)
            
            lab,plab=LD.TestLoadModel(bb,0.5) # cant run on thread
            print(LD.Owners_load[int(lab)])
            print(plab)
            
            
            if plab > final_plabel : 
                final_label_lt.append(int(lab))
                final_plabel_lt.append(plab)
                final_owner_lt.append(LD.Owners_load[int(lab)])
                final_sound_lt.append(LD.Sounds_load[int(lab)])
                if plab>= just4show : just4show=plab
            else :
                final_label_lt.append(-1)
                final_plabel_lt.append(-1)
                final_owner_lt.append("")
                final_sound_lt.append("")
                
#        print(final_label_lt)
        c=Counter(final_label_lt)
#        print(c.values())
#        print(c.keys())
        maxValue=-1
        maxKey=-1
        for key, value in c.items():
        #    print(key)
        #    print(value)
            if value>=maxValue :
                maxValue=value
                maxKey=key
                
#        print(maxValue)
#        print(maxKey)  
        if (( maxValue/len(final_label_lt) ) > 0.2) and ( maxKey > -1 ) : 
            final_label=maxKey
            final_plabel=just4show
            final_owner=LD.Owners_load[int(maxKey)]
            final_sound=LD.Sounds_load[int(maxKey)]
            print(final_label)
            print(final_plabel)
            print(final_owner)
            print(final_sound)
            for j in range(3) :
                self.testMics1.displayResult1(
                          disp=True ,
                          pics="IDSS.png",
                          final_label=final_label ,
                          final_plabel=final_plabel ,
                          final_owner=final_owner ,
                          final_sound=final_sound ,
                          message=""
                          ) 

        else :
            print("Don't know !")
            for j in range(3) :
                self.testMics1.displayResult1(
                          disp=True ,
                          pics="IDSS.png",
                          final_label=final_label ,
                          final_plabel=final_plabel ,
                          final_owner="Not able to pedict the result" ,
                          final_sound="Probability is not acceptable" 
                          )
            final_owner="Not able to pedict the result"
            final_sound="Probability is not acceptable" 
    
        print("****")   
        
        self.acc_result[self.acc_count]=final_label
        if  self.acc_count == 2 :
             if((self.acc_result[0]==self.acc_result[1])and(self.acc_result[1]==self.acc_result[2])) and (not(final_label==-1)) :
                 thr=threading.Thread(target=self.cloudIotThred, args=[final_label,final_plabel,final_owner,final_sound])
                 thr.start()
                 for j in range(3) :
                     self.testMics1.displayResult1(
                                  disp=True ,
                                  pics="IDSS.png",
                                  final_label=final_label ,
                                  final_plabel=final_plabel ,
                                  final_owner=final_owner ,
                                  final_sound=final_sound ,
                                  message=final_owner + " , " + final_sound,
                                  col=(0,255,145)
                                  )  
             else :
                 for j in range(3) :
                     self.testMics1.displayResult1(
                                  disp=True ,
                                  pics="IDSS.png",
                                  final_label=final_label ,
                                  final_plabel=final_plabel ,
                                  final_owner=final_owner ,
                                  final_sound=final_sound ,
                                  message="No Result !",
                                  col=(0,255,145)
                                  )              
             self.acc_result[0]=self.acc_result[1]    
             self.acc_result[1]=self.acc_result[2]   
             self.acc_count=1
        self.acc_count = self.acc_count +1
          
        
        
###global share_flag
#ct = CloudIOT_Class() 
#ct.googleSheet_init()
#ct.googleSheet_responseServerThread()
#print("exit")
##print(share_flag)
#
#Test=DLLivePredict_Class()
#Test.loadDLmodel()
##Test.signelLiveMics_whole(recordTime=0)
##Test.signelLiveMics_whole1(recordTime=10)
#
#thr=threading.Thread(target=Test.signelLiveMics_whole1, args=[10 , 500])
#thr.start()