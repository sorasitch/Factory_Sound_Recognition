# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:37:23 2020

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
import json

import threading

class ReadRecordSound_Class():
    
    
#    files_list = [] # all own obj 
    
    def __init__(self,
                 ):
        self.folder_path=".\\SNDPATH"
        self.files_list = [] # only own obj
        self.X_DATASET = []
        self.Y_DATASET = []
        self.OWNERS = []
        self.SOUNDS = []
        self.__X_DATASET = []
        self.__X_DATASET0 = []
        self.FRAMESIZE=0
        self.MODELOWNERS = []
        self.MODELSOUNDS = []
        self.MODELLABEL = []
        self.FRAMESZ = []
        self.MODELS = []
        self.__MODELOWNERS = []
        self.__MODELSOUNDS = []
        self.__MODELLABEL = []
        self.__FRAMESZ = []
        self.__MODELS = []
        
        
 
    def readfiles(self):
        files=""
        path=self.folder_path
        dirs = os.listdir(path)
#        print(dirs)
        
        for owners in dirs:
            pass
            self.OWNERS.append(owners)
            owners_path = path + "\\" + owners
            dirs1 = os.listdir(owners_path)
#            print(dirs1)
            
            for sounds in dirs1:
                pass
                self.SOUNDS.append(sounds)
                sounds_path = owners_path + "\\" + sounds
                dirs2 = os.listdir(sounds_path)
#                print(dirs2)

                for names in dirs2:
                    pass
                    files = sounds_path + "\\" + names
                    self.files_list.append(files)
                    self.Y_DATASET.append([self.OWNERS.index(owners),self.SOUNDS.index(sounds),owners,sounds,1])
#                    print(files)
        
        if(len(self.files_list)>0):
            pass
            return self.files_list
            
        return files
    
    def savePics(self , names=""):
        pass
        path=".\\SNDDEBUG1"
        if(not os.path.exists(path)):
            os.mkdir(path)
            
        names=names.replace(self.folder_path+"\\", "") 
        names=names.replace("\\", "_")
        names=names.replace(".wav", "")

        names=path+"\\"+str(names)
        
        return names

    def labelRecord(self , names=""):
        pass     
        names=names.replace(self.folder_path+"\\", "") 
        names=names.replace("\\", "")
        names=names.replace(".wav", "")

        names=path+"\\"+str(names)
        
        return names

    def readWAV(self , disp=False , output=False , lowpass=0 , name="" , nperseg=512):
        name_list = self.readfiles()
        for name in name_list :
            pass
            CHUNK = 1024
            # Read voice from the directory
            wf = wave.open(name, 'rb')
            #Create player
            p = pyaudio.PyAudio()
            # Get the parameters of the voice file
            FORMAT = p.get_format_from_width(wf.getsampwidth())
            CHANNELS = wf.getnchannels()
            RATE = wf.getframerate()
            print('FORMAT: {} \nCHANNELS: {} \nRATE: {}'.format(FORMAT, CHANNELS, RATE))
            
            if output : 
                # Open the audio stream, output=True for the audio output
                stream = p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                frames_per_buffer=CHUNK,
                                output=output)
            
            # play stream (3) Read audio data to the audio stream in 1024 blocks and play it
            frames = []
            frames1 = []
            o=0
            while (True):
                data = wf.readframes(CHUNK)
                if(len(data) <= 0):
                    break
            #    print(len(data))
#                if output : stream.write(data)
                frames.append(data)
                o=o+1
                
    #        print(o)
            
            ss=0
            for amp in frames :
                pass
                data1 = np.fromstring(amp,dtype=np.int16)
                frames1.extend(data1)
                ss=ss+1
                
            #show STFT and WAV
            amplitude=np.zeros([len(frames1)])
            amplitude[:] = frames1
            lf=lowpass
        
            if lf > 0 :
                b, a = signal.butter(10, lf, 'lp', fs=RATE)
    #            b, a = signal.iirfilter(10, [lf], btype='low',
    #                                   analog=False, ftype='bessel', fs=RATE)
                amplitude = signal.lfilter(b, a, amplitude)

            if output : 
                amplitude1=amplitude.reshape([ss,len(data1)]).astype(np.int16)
                for amp in amplitude1 :
                    amp1=amp.tobytes()
                    stream.write(amp1)
                    pass 
                stream.stop_stream()
                stream.close()
                p.terminate()
            wf.close()

#            amplitude = frames1
            #print(amplitude)
            f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
            Zxxlog = np.log(np.abs(Zxx)+1e-10)
            
            if lf > 0 :
                lfindex=np.where( f >= lf )
                idx=lfindex[0]
                Zxxlog=Zxxlog[:idx[0],:]
                f=f[:idx[0]]
    
            self.X_DATASET.append(Zxxlog)
    
            if disp :
                
                debugname=self.savePics(name)  
                
                print('f : ' , f.shape )
                print("t : " , t.shape )
                print("Zxx : " , Zxx.shape )
                
                fig = plt.figure()
                s = fig.add_subplot(111)
                s.plot(amplitude)
                fig.savefig(debugname+"_wav.png")
                plt.close(fig)
                img0 = cv2.imread(debugname+"_wav.png")
                cv2.imshow("SIGNAL",img0)
                #cv2.waitKey(2)
                
                fig = plt.figure(figsize=(14, 8))
                plt.pcolormesh(t, f, Zxxlog)
                plt.title('STFT Magnitude')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                #plt.show()
                fig.savefig(debugname+"_stft.png")
                plt.close(fig)
                img = cv2.imread(debugname+"_stft.png")
                cv2.imshow("STFT",img)
                cv2.waitKey(2)
                
                
        pass
        if disp :   
#            cv2.waitKey(0)
            cv2.destroyAllWindows()
                


    def maxFrameRecordWAV(self):
        size=0
        frame_size=0
        file_max=""
        path=self.folder_path
        dirs = os.listdir(path)
#        print(dirs)
        
        for owners in dirs:
            pass
            if owners == "ZZZ" : continue
            owners_path = path + "\\" + owners
            dirs1 = os.listdir(owners_path)
#            print(dirs1)
            
            for sounds in dirs1:
                pass
                sounds_path = owners_path + "\\" + sounds
                dirs2 = os.listdir(sounds_path)
#                print(dirs2)

                for names in dirs2:
                    pass
                    file = sounds_path + "\\" + names
                    sz = os.path.getsize(file) 
                    if sz > size : 
                        size=sz
                        file_max=file
         
        if file_max=="" : return 0            
        frame_size=self.returnFrameWAV(file_max)
        return frame_size
    
    
    def returnFrameWAV(self , name="" ):
        pass
        CHUNK = 1024
        # Read voice from the directory
        wf = wave.open(name, 'rb')
        #Create player
        p = pyaudio.PyAudio()
        # Get the parameters of the voice file
        FORMAT = p.get_format_from_width(wf.getsampwidth())
        CHANNELS = wf.getnchannels()
        RATE = wf.getframerate()
        print('FORMAT: {} \nCHANNELS: {} \nRATE: {}'.format(FORMAT, CHANNELS, RATE))
        
        # play stream (3) Read audio data to the audio stream in 1024 blocks and play it

        o=0
        while (True):
            data = wf.readframes(CHUNK)
            if(len(data) <= 0):
                break
        #    print(len(data))
            o=o+1
            
        wf.close()
        return o


    def readRecordWAV(self , disp=False , output=False , lowpass=0 , nperseg=512):
        files=""
        path=self.folder_path
        dirs = os.listdir(path)
#        print(dirs)
        
        for owners in dirs:
            pass
            self.OWNERS.append(owners)
            owners_path = path + "\\" + owners
            dirs1 = os.listdir(owners_path)
#            print(dirs1)
            
            self.__X_DATASET=[]
            for sounds in dirs1:
                pass
                self.SOUNDS.append(sounds)
                sounds_path = owners_path + "\\" + sounds
                dirs2 = os.listdir(sounds_path)
#                print(dirs2)

                self.__X_DATASET0=[]
                for names in dirs2:
                    pass
                    files = sounds_path + "\\" + names
                    self.__X_DATASET0.append(self.WAV(disp=disp , output=output , lowpass=lowpass , name=files , nperseg=nperseg )) #, owners=self.OWNERS.index(owners) , sounds=self.SOUNDS.index(sounds) )
                    self.files_list.append(files)
                    self.Y_DATASET.append([self.OWNERS.index(owners),self.SOUNDS.index(sounds),owners,sounds,1])
#                    print(files) 
                self.__X_DATASET.append(self.__X_DATASET0)
#                self.__X_DATASET.append([self.OWNERS.index(owners),self.SOUNDS.index(sounds),self.__X_DATASET0])   
                
            self.X_DATASET.append(self.__X_DATASET) 
#        self.X_DATASET.append([self.OWNERS.index(owners),self.SOUNDS.index(sounds),self.__X_DATASET]) 
        
        if(len(self.files_list)>0):
            pass
            return self.files_list
            
        return files


    def WAV(self , disp=False , output=False , lowpass=0 , name="" , nperseg=512 ):
        pass
        CHUNK = 1024
        # Read voice from the directory
        wf = wave.open(name, 'rb')
        #Create player
        p = pyaudio.PyAudio()
        # Get the parameters of the voice file
        FORMAT = p.get_format_from_width(wf.getsampwidth())
        CHANNELS = wf.getnchannels()
        RATE = wf.getframerate()
        print('FORMAT: {} \nCHANNELS: {} \nRATE: {}'.format(FORMAT, CHANNELS, RATE))
        
        if output : 
            # Open the audio stream, output=True for the audio output
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            frames_per_buffer=CHUNK,
                            output=output)
        
        # play stream (3) Read audio data to the audio stream in 1024 blocks and play it
        frames = []
        frames1 = []
        o=0
        while (True):
            data = wf.readframes(CHUNK)
            if(len(data) <= 0):
                break
        #    print(len(data))
#                if output : stream.write(data)
            frames.append(data)
            o=o+1
            
#        print(o)
        
        ss=0
        for amp in frames :
            pass
            data1 = np.fromstring(amp,dtype=np.int16)
            frames1.extend(data1)
            ss=ss+1
            
        #show STFT and WAV
        amplitude=np.zeros([len(frames1)])
        amplitude[:] = frames1
        lf=lowpass
    
        if lf > 0 :
            b, a = signal.butter(10, lf, 'lp', fs=RATE)
#            b, a = signal.iirfilter(10, [lf], btype='low',
#                                   analog=False, ftype='bessel', fs=RATE)
            amplitude = signal.lfilter(b, a, amplitude)

        if output : 
            amplitude1=amplitude.reshape([ss,len(data1)]).astype(np.int16)
            for amp in amplitude1 :
                amp1=amp.tobytes()
                stream.write(amp1)
                pass 
            stream.stop_stream()
            stream.close()
            p.terminate()
        wf.close()

#            amplitude = frames1
        #print(amplitude)
        f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
        Zxxlog = np.log(np.abs(Zxx)+1e-10)
        
        if lf > 0 :
            lfindex=np.where( f >= lf )
            idx=lfindex[0]
            Zxxlog=Zxxlog[:idx[0],:]
            f=f[:idx[0]]

#        self.__X_DATASET.append([owners,sounds,Zxxlog])

        if disp :
            
            debugname=self.savePics(name)  
            
            print('f : ' , f.shape )
            print("t : " , t.shape )
            print("Zxx : " , Zxx.shape )
            
            fig = plt.figure()
            s = fig.add_subplot(111)
            s.plot(amplitude)
            fig.savefig(debugname+"_wav.png")
            plt.close(fig)
            img0 = cv2.imread(debugname+"_wav.png")
            cv2.imshow("SIGNAL",img0)
            #cv2.waitKey(2)
            
            fig = plt.figure(figsize=(14, 8))
            plt.pcolormesh(t, f, Zxxlog)
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            #plt.show()
            fig.savefig(debugname+"_stft.png")
            plt.close(fig)
            img = cv2.imread(debugname+"_stft.png")
            cv2.imshow("STFT",img)
            cv2.waitKey(2)
            
            
        pass
        if disp :   
    #            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return Zxxlog
    
    
    def readRecordWAV1(self , disp=False , output=False , lowpass=0 , nperseg=512 ):
        
        size=self.maxFrameRecordWAV()
        self.FRAMESIZE=size
        print("FRAMESIZE : " + str(size))
        
         
        files=""
        path=self.folder_path
        dirs = os.listdir(path)
#        print(dirs)
        
        
        for owners in dirs:
            pass
            self.OWNERS.append(owners)
            owners_path = path + "\\" + owners
            dirs1 = os.listdir(owners_path)
            print(owners_path)
            
            self.__X_DATASET=[]
            for sounds in dirs1:
                pass
                self.SOUNDS.append(sounds)
                sounds_path = owners_path + "\\" + sounds
                dirs2 = os.listdir(sounds_path)
                print(sounds_path)

                self.__X_DATASET0=[]
                for names in dirs2:
                    pass
                    files = sounds_path + "\\" + names
                    self.__X_DATASET0.append(self.WAV1(disp=disp , output=output , lowpass=lowpass , name=files , nperseg=nperseg , size=size )) #, owners=self.OWNERS.index(owners) , sounds=self.SOUNDS.index(sounds) )
                    self.files_list.append(files)
                    self.Y_DATASET.append([self.OWNERS.index(owners),self.SOUNDS.index(sounds),owners,sounds,1])
#                    print(files) 
                self.__X_DATASET.append(self.__X_DATASET0)
#                self.__X_DATASET.append([self.OWNERS.index(owners),self.SOUNDS.index(sounds),self.__X_DATASET0])   
                
            self.X_DATASET.append(self.__X_DATASET) 
#        self.X_DATASET.append([self.OWNERS.index(owners),self.SOUNDS.index(sounds),self.__X_DATASET]) 

        # if disp :   
    # #            cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        if(len(self.files_list)>0):
            pass
            return self.files_list
            
        return files
    

    def WAV1(self , disp=False , output=False , lowpass=0 , name="" , nperseg=512 , size=0 ):
        pass
        CHUNK = 1024
        # Read voice from the directory
        wf = wave.open(name, 'rb')
        #Create player
        p = pyaudio.PyAudio()
        # Get the parameters of the voice file
        FORMAT = p.get_format_from_width(wf.getsampwidth())
        CHANNELS = wf.getnchannels()
        RATE = wf.getframerate()
        print('FORMAT: {} \nCHANNELS: {} \nRATE: {}'.format(FORMAT, CHANNELS, RATE))
        
        if output : 
            # Open the audio stream, output=True for the audio output
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            frames_per_buffer=CHUNK,
                            output=output)
        
        # play stream (3) Read audio data to the audio stream in 1024 blocks and play it
        frames = []
        frames1 = []
        o=0
        while (True):
            data = wf.readframes(CHUNK)
            if(len(data) <= 0):
                break
        #    print(len(data))
#                if output : stream.write(data)
            frames.append(data)
            o=o+1
            
#        print(o)
        
        ss=0
        for amp in frames :
            pass
            data1 = np.fromstring(amp,dtype=np.int16)
            frames1.extend(data1)
            ss=ss+1
         
        if size>0 :
            fr_size=ss
            ad_size=0
            if fr_size<size :
                ad_size=size-fr_size
                ad_data=[0 for x in range(ad_size*CHUNK*2)]
#               print("frame1 0: " + str(len(frames1)))
                frames1.extend(ad_data)
                ss=ss+ad_size
#               print("frame1 1: " + str(len(frames1)))
            if fr_size>size :
                ad_size=fr_size-size
                frames1=frames1[0:(size*CHUNK*2)]
                ss=size
#               print("frame1 0: " + str(len(frames1)))

            
        
            
        #show STFT and WAV
        amplitude=np.zeros([len(frames1)])
        amplitude[:] = frames1
        lf=lowpass
    
        if lf > 0 :
            b, a = signal.butter(10, lf, 'lp', fs=RATE)
#            b, a = signal.iirfilter(10, [lf], btype='low',
#                                   analog=False, ftype='bessel', fs=RATE)
            amplitude = signal.lfilter(b, a, amplitude)

        if output : 
            amplitude1=amplitude.reshape([ss,len(data1)]).astype(np.int16)
            for amp in amplitude1 :
                amp1=amp.tobytes()
                stream.write(amp1)
                pass 
            stream.stop_stream()
            stream.close()
            p.terminate()
        wf.close()

#            amplitude = frames1
        #print(amplitude)
        f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
        Zxxlog = np.log(np.abs(Zxx)+1e-10)
        
        if lf > 0 :
            lfindex=np.where( f >= lf )
            idx=lfindex[0]
            Zxxlog=Zxxlog[:idx[0],:]
            f=f[:idx[0]]

#        self.__X_DATASET.append([owners,sounds,Zxxlog])

        if disp :
            
            debugname=self.savePics(name)  
            
            print('f : ' , f.shape )
            print("t : " , t.shape )
            print("Zxx : " , Zxx.shape )
            
            fig = plt.figure()
            s = fig.add_subplot(111)
            s.plot(amplitude)
            fig.savefig(debugname+"_wav.png")
            plt.close(fig)
            img0 = cv2.imread(debugname+"_wav.png")
            cv2.imshow("SIGNAL",img0)
            #cv2.waitKey(2)
            
            fig = plt.figure(figsize=(14, 8))
            plt.pcolormesh(t, f, Zxxlog)
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            #plt.show()
            fig.savefig(debugname+"_stft.png")
            plt.close(fig)
            img = cv2.imread(debugname+"_stft.png")
            cv2.imshow("STFT",img)
            cv2.waitKey(2)
            

        if disp :   
    #            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return Zxxlog
    
    
    def signelWAV1(self , name="" ):
        pass
        CHUNK = 1024
        # Read voice from the directory
        wf = wave.open(name, 'rb')
        #Create player
        p = pyaudio.PyAudio()
        # Get the parameters of the voice file
        FORMAT = p.get_format_from_width(wf.getsampwidth())
        CHANNELS = wf.getnchannels()
        RATE = wf.getframerate()
        print('FORMAT: {} \nCHANNELS: {} \nRATE: {}'.format(FORMAT, CHANNELS, RATE))
        
        # play stream (3) Read audio data to the audio stream in 1024 blocks and play it
        frames = []
        frames1 = []
        o=0
        while (True):
            data = wf.readframes(CHUNK)
            if(len(data) <= 0):
                break
        #    print(len(data))
#                if output : stream.write(data)
            frames.append(data)
            o=o+1
            
#        print(o)
        wf.close()
        
        ss=0
        for amp in frames :
            pass
            data1 = np.fromstring(amp,dtype=np.int16)
            frames1.extend(data1)
            ss=ss+1

        frames_data=frames1
        frame_size=ss
        dataPerFrame=len(data1)
        return frames_data,frame_size,dataPerFrame,FORMAT,CHANNELS,RATE
 


    def spectrogramWAV1(self , disp=False , output=False , lowpass=0 , name="" , nperseg=512 , size=0 , frames_data=[] , frame_size=0 , dataPerFrame=0 , FORMAT=pyaudio.paInt16, CHANNELS=2 ,RATE=44100 ):

        frames1=[]
#        frames1=frames_data #pionter 
        frames1.extend(frames_data)
        ss=0
        ss=frame_size
        
        CHUNK = 1024
        FORMAT=FORMAT
        CHANNELS=CHANNELS
        RATE=RATE
        
        p = pyaudio.PyAudio()
        
        if output : 
            # Open the audio stream, output=True for the audio output
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            frames_per_buffer=CHUNK,
                            output=output)
            

        if size>0 :
            fr_size=ss
            ad_size=0
            if fr_size<size :
                ad_size=size-fr_size
                ad_data=[0 for x in range(ad_size*CHUNK*2)]
#                print("frame1 0: " + str(len(frames1)))
                frames1.extend(ad_data)
                ss=ss+ad_size
#                print("frame1 1: " + str(len(frames1)))
            if fr_size>size :
                ad_size=fr_size-size
                frames1=frames1[0:(size*CHUNK*2)]
                ss=size
#                print("frame1 0: " + str(len(frames1)))
            
        
        #show STFT and WAV
        amplitude=np.zeros([len(frames1)])
        amplitude[:] = frames1
        lf=lowpass
    
        if lf > 0 :
            b, a = signal.butter(10, lf, 'lp', fs=RATE)
#            b, a = signal.iirfilter(10, [lf], btype='low',
#                                   analog=False, ftype='bessel', fs=RATE)
            amplitude = signal.lfilter(b, a, amplitude)

        if output : 
            amplitude1=amplitude.reshape([ss,dataPerFrame]).astype(np.int16)
            for amp in amplitude1 :
                amp1=amp.tobytes()
                stream.write(amp1)
                pass 
            stream.stop_stream()
            stream.close()
            p.terminate()

        
#            amplitude = frames1
        #print(amplitude)
        f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
        Zxxlog = np.log(np.abs(Zxx)+1e-10)
        
        if lf > 0 :
            lfindex=np.where( f >= lf )
            idx=lfindex[0]
            Zxxlog=Zxxlog[:idx[0],:]
            f=f[:idx[0]]

#        self.__X_DATASET.append([owners,sounds,Zxxlog])

        if disp :
            
            debugname=self.savePics(name)   
            
            print('f : ' , f.shape )
            print("t : " , t.shape )
            print("Zxx : " , Zxx.shape )
            
            fig = plt.figure()
            s = fig.add_subplot(111)
            s.plot(amplitude)
            fig.savefig(debugname+"_wav.png")
            plt.close(fig)
            img0 = cv2.imread(debugname+"_wav.png")
            cv2.imshow("SIGNAL",img0)
            #cv2.waitKey(2)
            
            fig = plt.figure(figsize=(14, 8))
            plt.pcolormesh(t, f, Zxxlog)
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            #plt.show()
            fig.savefig(debugname+"_stft.png")
            plt.close(fig)
            img = cv2.imread(debugname+"_stft.png")
            cv2.imshow("STFT",img)
            cv2.waitKey(2)
            
            
        pass
        if disp :   
    #            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return Zxxlog
    
    
    def saveModelName(self , owners="" , sounds=""):
        pass
        path=".\\SNDMODEL"
        if(not os.path.exists(path)):
            os.mkdir(path)       
            
        names = path+"\\"+str(owners)+"_"+str(sounds)

        return names          
    
    
    def saveModelNameWithData(self , owners="" , sounds="" , data=""):
        pass
        path=".\\SNDMODEL"
        if(not os.path.exists(path)):
            os.mkdir(path)       
        jsonpath=".\\SNDJSON"
        if(not os.path.exists(jsonpath)):
            os.mkdir(jsonpath)
            
        names = path+"\\"+str(owners)+"_"+str(sounds)
        
        with open(jsonpath + "\\"+str(owners)+"_"+str(sounds) + ".txt", 'w') as outfile:
            json.dump(data, outfile)
        
        return names 
    
    
    def readModel(self):
        path=".\\SNDMODEL"
        dirs = os.listdir(path)
#        print(dirs)
        for models in dirs:
            pass
            names=models.replace(".hdf51", "") 
            names=names.split("_")  #[owners,sounds]
            self.MODELOWNERS.append(names[0])
            self.MODELSOUNDS.append(names[1])
            self.MODELS.append(path + "\\" + names)
    
        return self.MODELS
    
    def readModelWithData(self):
        path=".\\SNDMODEL"
        jsonpath=".\\SNDJSON"
        dirs = os.listdir(path)
#        print(dirs)
        self.__MODELS = []
        for models in dirs:
            pass
        
            if models.find(".hdf51") <= 0 : continue
        
            names=models.replace(".hdf51", "") 
            jsonnames=names.replace(path+"\\", "") 
            
            with open(jsonpath + "\\" + jsonnames + ".txt") as json_file:
                data = json.load(json_file)
                
#            print(data)  
            
            da=np.array(data)
            self.__MODELOWNERS = []
            self.__MODELSOUNDS = []
            self.__MODELLABEL = []
            self.__FRAMESZ = []
            
            for d in da :
                self.__MODELOWNERS.append(d[1])
                self.__MODELSOUNDS.append(d[2])
                self.__MODELLABEL.append(int(d[0]))
                self.__FRAMESZ.append(int(d[3]))
                
            self.MODELS.append(path + "\\" + names)
            self.MODELOWNERS.append(self.__MODELOWNERS)
            self.MODELSOUNDS.append(self.__MODELSOUNDS)
            self.MODELLABEL.append(self.__MODELLABEL)
            self.FRAMESZ.append(self.__FRAMESZ)
            
        return self.MODELS
  
    
    def spectrogramWAV2(self , disp=False , output=False , lowpass=0 , name="" , nperseg=512 , size=0 , frames_data=[] , frame_size=0 , dataPerFrame=0 , FORMAT=pyaudio.paInt16, CHANNELS=2 ,RATE=44100 ):

        frames1=[]
#        frames1=frames_data #pionter 
        frames1.extend(frames_data)
        ss=0
        ss=frame_size
        
        CHUNK = 1024
        FORMAT=FORMAT
        CHANNELS=CHANNELS
        RATE=RATE
        
        p = pyaudio.PyAudio()
        
        if output : 
            # Open the audio stream, output=True for the audio output
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            frames_per_buffer=CHUNK,
                            output=output)
            

        if size>0 :
            fr_size=ss
            ad_size=0
            if fr_size<size :
                ad_size=size-fr_size
                ad_data=[0 for x in range(ad_size*CHUNK*2)]
#                print("frame1 0: " + str(len(frames1)))
                frames1.extend(ad_data)
                ss=ss+ad_size
#                print("frame1 1: " + str(len(frames1)))
            if fr_size>size :
                ad_size=fr_size-size
                frames1=frames1[0:(size*CHUNK*2)]
                ss=size
#                print("frame1 0: " + str(len(frames1)))
            
        
        #show STFT and WAV
        amplitude=np.zeros([len(frames1)])
        amplitude[:] = frames1
        lf=lowpass
    
        if lf > 0 :
            b, a = signal.butter(10, lf, 'lp', fs=RATE)
#            b, a = signal.iirfilter(10, [lf], btype='low',
#                                   analog=False, ftype='bessel', fs=RATE)
            amplitude = signal.lfilter(b, a, amplitude)

        if output : 
            amplitude1=amplitude.reshape([ss,dataPerFrame]).astype(np.int16)
            for amp in amplitude1 :
                amp1=amp.tobytes()
                stream.write(amp1)
                pass 
            stream.stop_stream()
            stream.close()
            p.terminate()

        
#            amplitude = frames1
        #print(amplitude)
        f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
        Zxxlog = np.log(np.abs(Zxx)+1e-10)
        
        if lf > 0 :
            lfindex=np.where( f >= lf )
            idx=lfindex[0]
            Zxxlog=Zxxlog[:idx[0],:]
            f=f[:idx[0]]

#        self.__X_DATASET.append([owners,sounds,Zxxlog])


        if disp :
            pass
            thr=threading.Thread(target=self.displaySpectrogramWAV2, args=[name,amplitude,Zxxlog,f,t,Zxx])
            thr.start()
#            self.displaySpectrogramWAV2(name,amplitude,Zxxlog,f,t,Zxx)



#        if disp :   
#    #            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
        return Zxxlog
    
    
    def displaySpectrogramWAV2(self , 
                               name="",
                               amplitude=[],
                               Zxxlog=[],
                               f=[],
                               t=[],
                               Zxx=[]
                               ):

            
        debugname=self.savePics(name)   
        
        print('f : ' , f.shape )
        print("t : " , t.shape )
        print("Zxx : " , Zxx.shape )
        
        fig = plt.figure()
        s = fig.add_subplot(111)
        s.plot(amplitude)
        fig.savefig(debugname+"_wav.png")
        plt.close(fig)
        img0 = cv2.imread(debugname+"_wav.png")
        cv2.imshow("SIGNAL",img0)
        #cv2.waitKey(2)
        
        fig = plt.figure(figsize=(14, 8))
        plt.pcolormesh(t, f, Zxxlog)
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        #plt.show()
        fig.savefig(debugname+"_stft.png")
        plt.close(fig)
        img = cv2.imread(debugname+"_stft.png")
        cv2.imshow("STFT",img)
        cv2.waitKey(2)
            
    def destroydisplay(self):
        cv2.destroyAllWindows()
    
    
    def compressedSptg(self,
                       f=[],
                       t=[],
                       Zxx=[],
                       dv=100
                       ):
        z=np.abs(Zxx)
        fx=f
        t=t
        y=z.shape[0]
#        x=z.shape[1]
        dv=dv
        yr=y/dv
        
        frq=[]
        fc=[]
        for i in range(int(yr)):
            frq.append(np.sum(z[i*dv+0:i*dv+dv,:],axis=0))
            fc.append(np.sum(fx[i*dv+0:i*dv+dv])/dv)
            
        sptg=np.array(frq).astype('float')
        fc=np.array(fc).astype('int')
        print(sptg.shape)
#        print(fc.shape)

        return fc,t,sptg
    
    def readRecordWAV1compress(self , disp=False , output=False , lowpass=0 , nperseg=512 , compressed=0 , logScl=True ):
        
        size=self.maxFrameRecordWAV()
        self.FRAMESIZE=size
        print("FRAMESIZE : " + str(size))
        
         
        files=""
        path=self.folder_path
        dirs = os.listdir(path)
#        print(dirs)
        
        
        for owners in dirs:
            pass
            self.OWNERS.append(owners)
            owners_path = path + "\\" + owners
            dirs1 = os.listdir(owners_path)
            print(owners_path)
            
            self.__X_DATASET=[]
            for sounds in dirs1:
                pass
                self.SOUNDS.append(sounds)
                sounds_path = owners_path + "\\" + sounds
                dirs2 = os.listdir(sounds_path)
                print(sounds_path)

                self.__X_DATASET0=[]
                for names in dirs2:
                    pass
                    files = sounds_path + "\\" + names
                    self.__X_DATASET0.append(self.WAV1compress(disp=disp , output=output , lowpass=lowpass , name=files , nperseg=nperseg , size=size , compressed=compressed , logScl=logScl )) #, owners=self.OWNERS.index(owners) , sounds=self.SOUNDS.index(sounds) )
                    self.files_list.append(files)
                    self.Y_DATASET.append([self.OWNERS.index(owners),self.SOUNDS.index(sounds),owners,sounds,1])
#                    print(files) 
                self.__X_DATASET.append(self.__X_DATASET0)
#                self.__X_DATASET.append([self.OWNERS.index(owners),self.SOUNDS.index(sounds),self.__X_DATASET0])   
                
            self.X_DATASET.append(self.__X_DATASET) 
#        self.X_DATASET.append([self.OWNERS.index(owners),self.SOUNDS.index(sounds),self.__X_DATASET]) 

        # if disp :   
    # #            cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        if(len(self.files_list)>0):
            pass
            return self.files_list
            
        return files
    

    def WAV1compress(self , disp=False , output=False , lowpass=0 , name="" , nperseg=512 , size=0 , compressed=0 , logScl=True):
        pass
        CHUNK = 1024
        # Read voice from the directory
        wf = wave.open(name, 'rb')
        #Create player
        p = pyaudio.PyAudio()
        # Get the parameters of the voice file
        FORMAT = p.get_format_from_width(wf.getsampwidth())
        CHANNELS = wf.getnchannels()
        RATE = wf.getframerate()
        print('FORMAT: {} \nCHANNELS: {} \nRATE: {}'.format(FORMAT, CHANNELS, RATE))
        
        if output : 
            # Open the audio stream, output=True for the audio output
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            frames_per_buffer=CHUNK,
                            output=output)
        
        # play stream (3) Read audio data to the audio stream in 1024 blocks and play it
        frames = []
        frames1 = []
        o=0
        while (True):
            data = wf.readframes(CHUNK)
            if(len(data) <= 0):
                break
        #    print(len(data))
#                if output : stream.write(data)
            frames.append(data)
            o=o+1
            
#        print(o)
        
        ss=0
        for amp in frames :
            pass
            data1 = np.fromstring(amp,dtype=np.int16)
            frames1.extend(data1)
            ss=ss+1
         
        if size>0 :
            fr_size=ss
            ad_size=0
            if fr_size<size :
                ad_size=size-fr_size
                ad_data=[0 for x in range(ad_size*CHUNK*2)]
#               print("frame1 0: " + str(len(frames1)))
                frames1.extend(ad_data)
                ss=ss+ad_size
#               print("frame1 1: " + str(len(frames1)))
            if fr_size>size :
                ad_size=fr_size-size
                frames1=frames1[0:(size*CHUNK*2)]
                ss=size
#               print("frame1 0: " + str(len(frames1)))

            
        
            
        #show STFT and WAV
        amplitude=np.zeros([len(frames1)])
        amplitude[:] = frames1
        lf=lowpass
    
        if lf > 0 :
            b, a = signal.butter(10, lf, 'lp', fs=RATE)
#            b, a = signal.iirfilter(10, [lf], btype='low',
#                                   analog=False, ftype='bessel', fs=RATE)
            amplitude = signal.lfilter(b, a, amplitude)

        if output : 
            amplitude1=amplitude.reshape([ss,len(data1)]).astype(np.int16)
            for amp in amplitude1 :
                amp1=amp.tobytes()
                stream.write(amp1)
                pass 
            stream.stop_stream()
            stream.close()
            p.terminate()
        wf.close()

#            amplitude = frames1
        #print(amplitude)
        f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
        
        if compressed <= 0 :
            Zxxlog = np.abs(Zxx)+1e-10
            if logScl : Zxxlog = np.log(Zxxlog)
            print(Zxx.shape )
            
        else :
            f, t, ZxxAbs = self.compressedSptg(
                           f=f,
                           t=t,
                           Zxx=Zxx,
                           dv=compressed
                           )
            Zxxlog = ZxxAbs+1e-10
            if logScl : Zxxlog = np.log(Zxxlog)
        
        if lf > 0 :
            lfindex=np.where( f >= lf )
            idx=lfindex[0]
            Zxxlog=Zxxlog[:idx[0],:]
            f=f[:idx[0]]

#        self.__X_DATASET.append([owners,sounds,Zxxlog])

        if disp :
            
            debugname=self.savePics(name)  
            
            print('f : ' , f.shape )
            print("t : " , t.shape )
            print("Zxx : " , Zxx.shape )
            
            fig = plt.figure()
            s = fig.add_subplot(111)
            s.plot(amplitude)
            fig.savefig(debugname+"_wav.png")
            plt.close(fig)
            img0 = cv2.imread(debugname+"_wav.png")
            cv2.imshow("SIGNAL",img0)
            #cv2.waitKey(2)
            
            fig = plt.figure(figsize=(14, 8))
            plt.pcolormesh(t, f, Zxxlog)
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            #plt.show()
            fig.savefig(debugname+"_stft.png")
            plt.close(fig)
            img = cv2.imread(debugname+"_stft.png")
            cv2.imshow("STFT",img)
            cv2.waitKey(2)
            

        if disp :   
    #            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return Zxxlog
#
#WAV=ReadRecordSound_Class() 
##WAV.readWAV(disp=False , output=False , lowpass=0)
##print(WAV.Y_DATASET)
##print(WAV.OWNERS)
##print(WAV.SOUNDS)
##print(WAV.X_DATASET)
##y = np.array(WAV.Y_DATASET)
##print(y[:,3])
##print(y[:,4])
##print(to_categorical(y[:,4], num_classes = 2))
##print(y.shape)
##x = np.array(WAV.X_DATASET)
##print(x.shape)
#
#WAV.readRecordWAV(disp=False , output=False , lowpass=0)
##print(WAV.Y_DATASET)
##print(WAV.OWNERS)
##print(WAV.SOUNDS)
##print(WAV.X_DATASET)
#y = np.array(WAV.Y_DATASET)
##print(y[:,2])
##print(y[:,3])
##print(y[:,4])
##print(to_categorical(y[:,4], num_classes = 2))
##print(y.shape)
#x = np.array(WAV.X_DATASET)
#print("****") 
#print(x.shape) 
##print(x[0].shape)
##print(x[0][0].shape)
#idx_owners=0
#for owners in x:
##    print(owners.shape)
#    print(WAV.OWNERS[idx_owners])
#    idx_sounds=0
#    for sounds in owners:
##        print(sounds.shape)  #used
#        print(WAV.SOUNDS[idx_sounds])
#        #train deep lerning here
#        x_train = sounds
#        y_train = to_categorical(np.ones((sounds.shape[0],), dtype=int), num_classes = 2)
#        model_name=WAV.saveModelName(owners=WAV.OWNERS[idx_owners] , sounds=WAV.SOUNDS[idx_sounds])
#        
#        print(x_train.shape)
#        print(y_train.shape)
#        print(model_name)
#        
#        for files in sounds:
#            print(files.shape)
#            
#        idx_sounds=idx_sounds+1
#            
#    idx_owners=idx_owners+1   
#print("****") 
#WAV.readModel()
#for owners,sounds,models in zip(WAV.MODELOWNERS,WAV.MODELSOUNDS,WAV.MODELS) :
#    print(owners)
#    print(sounds)
#    print(models)
#print("****") 
#
