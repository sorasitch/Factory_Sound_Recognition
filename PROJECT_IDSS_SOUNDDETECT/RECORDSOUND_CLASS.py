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

from sys import byteorder
from array import array
from struct import pack

import threading
import time
import datetime

class RecoedSound_Class():
    
    
    def __init__(self,
                 owners="",
                 sounds="",
                 ):
        self.folder_path=".\\SNDPATH"
        self.owners=owners
        self.sounds=sounds
        self.names=""
        self.__files=""
        
        
    def savefiles(self):
        pass
        path=self.folder_path
        if(not os.path.exists(path)):
            os.mkdir(path)
    
        owners_path=path+"\\"+self.owners
        if(not os.path.exists(owners_path)):
            os.mkdir(owners_path)
         
        sounds_path=owners_path+"\\"+self.sounds
        if(not os.path.exists(sounds_path)):
            os.mkdir(sounds_path)
        
        dirs = os.listdir(sounds_path)
        
        n=0
        for names in dirs:
            nn=int(names.replace(".wav", ""))
            if n < int(nn) :
                n=int(nn)
            
        n=n+1
        self.__files=sounds_path+"\\"+str(n)+".wav"
        self.names=sounds_path+"\\"+str(n)
        
        
        return self.__files
    
    def savePics(self):
        pass
        path=".\\SNDDEBUG"
        if(not os.path.exists(path)):
            os.mkdir(path)
    
        owners_path=path+"\\"+self.owners
        if(not os.path.exists(owners_path)):
            os.mkdir(owners_path)
         
        sounds_path=owners_path+"\\"+self.sounds
        if(not os.path.exists(sounds_path)):
            os.mkdir(sounds_path)
        
        dirs = os.listdir(sounds_path)
        
        n=0
        for names in dirs:
            if not ("_wav.png" in names):
                continue;
            nn=int(names.replace("_wav.png", ""))
            if n < int(nn) :
                n=int(nn)
            
        n=n+1

        names=sounds_path+"\\"+str(n)
        
        return names
        
        
    def Mics(self, disp=False , recordTime=10 , nperseg=512 , ZERO=False):  
        pass
        name=self.savefiles()
    
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = int(recordTime/2)  #5s record time
        WAVE_OUTPUT_FILENAME = name
          
        p = pyaudio.PyAudio()
#        print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print("* start recording")

        frames = []
        frames1 = []
        
        for i in range(0, int((RATE / CHUNK) * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            
            if ZERO :
#                zero=[0 for x in range(CHUNK*2)]
#                zero=np.array(zero).astype(np.int16)
#                zero=zero.astype(np.int16)
                zero=np.zeros([CHUNK*2]).astype(np.int16)
                zero1=zero.tobytes()
                data=zero1
#                print(data)
            frames.append(data)

        
        print("* done recording")
#        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if disp :
            
            debugname=self.savePics()
            
            ss=0
            for amp in frames :
                pass
                data1 = np.fromstring(amp,dtype=np.int16)
                frames1.extend(data1)
                ss=ss+1
                
            #show STFT and WAV
            amplitude = frames1
            #print(amplitude)
            f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
            print('f : ' , f.shape )
            print("t : " , t.shape )
            print("Zxx : " , Zxx.shape )
            
            fig = plt.figure()
            s = fig.add_subplot(111)
            s.plot(amplitude)
            fig.savefig(debugname+"_wav.png")
#            plt.close(fig)
            img0 = cv2.imread(debugname+"_wav.png")
            cv2.imshow("SIGNAL",img0)
#            cv2.waitKey(1)
            
            fig = plt.figure(figsize=(14, 8))
            plt.pcolormesh(t, f, np.log(np.abs(Zxx)+1e-10))
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            #plt.show()
            fig.savefig(debugname+"_stft.png")
            plt.close(fig)
            img = cv2.imread(debugname+"_stft.png")
            cv2.imshow("STFT",img)
            cv2.waitKey(2)
            
#            cv2.waitKey(0)
            cv2.destroyAllWindows()
     
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
  
    
    def Mics_section(self, disp=False , save=True ,recordTime=10 , lowpass=0 ,nperseg=512 , sammeword=False ): 
        pass
        name=self.savefiles()
    
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = int(recordTime/2)  #5s record time
        WAVE_OUTPUT_FILENAME = name
          
        p = pyaudio.PyAudio()
#        print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        

        frames = []
        frames0 = []
        
#        for i in range(0, int((RATE / CHUNK) * RECORD_SECONDS)):
        frameNumber=int((RATE / CHUNK) * RECORD_SECONDS)
        frameNoise=int((RATE / CHUNK) * 1.5) #3s
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
            
        print("* start recording")
        f=0
        while True:
            data = stream.read(CHUNK)
            data_chunk=array('h',data)
            vol=max(data_chunk)
            if(vol>=Noise):
                print("something said " + str(f))
                frames.append(data)
                snd_start=True
            else:
                pass
                if snd_start : 
                    frames0.append(frames)
                    frames = []
                    snd_start=False
#                print("nothing")
#            print("\n")
            if(f>=frameNumber): break
            f=f+1
        
        print("* done recording")
        
        if(len(frames0)==0) : return
        
#        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        Zxxlog=[]
        
        avg=0
        num=0
        if sammeword : 
            frm_len = []
            for frame_section in frames0 :
                frm_len.append(len(frame_section))
                num=num+1
            avg=max(frm_len)*(0.7)
 

        
        for frame_section in frames0 :
            pass
            ss=0
            frames1 = [] 
            
            if(len(frame_section)<avg): continue

            if disp :

                for amp in frame_section :
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
                
              
        #        amplitude = frames1
                #print(amplitude)
                f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
                Zxxlog = np.log(np.abs(Zxx)+1e-10)
                
                if lf > 0 :
                    lfindex=np.where( f >= lf )
                    idx=lfindex[0]
                    Zxxlog=Zxxlog[:idx[0],:]
                    f=f[:idx[0]]
        
                    debugname=self.savePics()
                    
                    print('f : ' , f.shape )
                    print("t : " , t.shape )
                    print("Zxxlog : " , Zxxlog.shape )
                    
                    fig = plt.figure()
                    s = fig.add_subplot(111)
                    s.plot(amplitude)
                    fig.savefig(debugname+"_wav.png")
                    plt.close(fig)
                    img0 = cv2.imread(debugname+"_wav.png")
                    cv2.imshow("SIGNAL",img0)
#                    cv2.waitKey(1)
                    
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
                
            if save :
                WAVE_OUTPUT_FILENAME=self.savefiles()
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frame_section))
                wf.close()
                
        if disp :   
    #            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
        
    def Mics_whole(self, disp=False , save=True  ,recordTime=10 , lowpass=0 ,nperseg=512):  
        pass
        name=self.savefiles()
    
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = int(recordTime/2)  #5s record time
        WAVE_OUTPUT_FILENAME = name
          
        p = pyaudio.PyAudio()
#        print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        frames0 = []
        
#        for i in range(0, int((RATE / CHUNK) * RECORD_SECONDS)):
        frameNumber=int((RATE / CHUNK) * RECORD_SECONDS)
        frameNoise=int((RATE / CHUNK) * 3) #3s
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
            
        print("* start recording")
        f=0
        while True:
            data = stream.read(CHUNK)
            data_chunk=array('h',data)
            vol=max(data_chunk)
            if(vol>=Noise):
                print("something said " + str(f))
                frames.append(data)
                snd_start=True
            else:
                pass
                if snd_start : 
                    frames0.append(frames)
                    frames = []
                    snd_start=False
#                print("nothing")
#            print("\n")
            if(f>=frameNumber): break
            f=f+1
        
        print("* done recording")
        
        if(len(frames0)==0) : return
        
#        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        Zxxlog=[]
        
        frames00 = []       
        for frame_section in frames0 :
            pass
            frames00.extend(frame_section)
        
        if disp :   
            
            ss=0
            frames1=[]
            for amp in frames00 :
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
            
    
    #        amplitude = frames1
            #print(amplitude)
            f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
            Zxxlog = np.log(np.abs(Zxx)+1e-10)
            
            if lf > 0 :
                lfindex=np.where( f >= lf )
                idx=lfindex[0]
                Zxxlog=Zxxlog[:idx[0],:]
                f=f[:idx[0]]
    
                debugname=self.savePics()
                
                print('f : ' , f.shape )
                print("t : " , t.shape )
                print("Zxxlog : " , Zxxlog.shape )
                
                fig = plt.figure()
                s = fig.add_subplot(111)
                s.plot(amplitude)
                fig.savefig(debugname+"_wav.png")
                plt.close(fig)
                img0 = cv2.imread(debugname+"_wav.png")
                cv2.imshow("SIGNAL",img0)
#                cv2.waitKey(1)
                
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
            
    
        if save : 
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames00))
            wf.close()
    

    
    def saveTestFiles(self):
        pass
        path=".\\SNDTESTDEBUG"
        if(not os.path.exists(path)):
            os.mkdir(path)
    
        owners_path=path+"\\"+self.owners
        if(not os.path.exists(owners_path)):
            os.mkdir(owners_path)
         
        sounds_path=owners_path+"\\"+self.sounds
        if(not os.path.exists(sounds_path)):
            os.mkdir(sounds_path)
        
        dirs = os.listdir(sounds_path)
        
        n=0
        for names in dirs:
            
            if names.find(".wav") <= 0 : continue
            
            nn=int(names.replace(".wav", ""))
            if n < int(nn) :
                n=int(nn)
            
        n=n+1
        self.__files=sounds_path+"\\"+str(n)+".wav"
        self.names=sounds_path+"\\"+str(n)
        
        
        return self.__files
    
    def saveTestPics(self):
        pass
        path=".\\SNDTESTDEBUG"
        if(not os.path.exists(path)):
            os.mkdir(path)
    
        owners_path=path+"\\"+self.owners
        if(not os.path.exists(owners_path)):
            os.mkdir(owners_path)
         
        sounds_path=owners_path+"\\"+self.sounds
        if(not os.path.exists(sounds_path)):
            os.mkdir(sounds_path)
        
        dirs = os.listdir(sounds_path)
        
        n=0
        for names in dirs:
            
            if names.find(".png") <= 0 : continue
            
            if not ("_wav.png" in names):
                continue;
            nn=int(names.replace("_wav.png", ""))
            if n < int(nn) :
                n=int(nn)
            
        n=n+1

        names=sounds_path+"\\"+str(n)
        
        return names
        

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
            
            debugname=self.saveTestPics() 
            
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
            #cv2.waitKey(1)
            
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
            
            debugname=self.saveTestPics()
            
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
            #cv2.waitKey(1)
            
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



    def spectrogramWAV1(self , disp=False , output=False , lowpass=0 , nperseg=512 , size=0 , frames_data=[] , frame_size=0 , dataPerFrame=0 , FORMAT=pyaudio.paInt16, CHANNELS=2 ,RATE=44100 ):

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
            print(amplitude.shape)
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
            
            debugname=self.saveTestPics()  
            
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
            #cv2.waitKey(1)
            
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
            
#        print("spectrogramWAV1::frames_data " + str(len(frames_data)))
        return Zxxlog
    
    
    
        
    def TestMics(self, disp=False , save=False ,recordTime=10 , lowpass=0 ,nperseg=512 ):  
        pass
        name=self.saveTestFiles()
    
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = int(recordTime/2)  #5s record time
        WAVE_OUTPUT_FILENAME = name
          
        p = pyaudio.PyAudio()
#        print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print("* start recording")

        frames = []
        frames1 = []
        
        for i in range(0, int((RATE / CHUNK) * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("* done recording")
#        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        
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
        
        
#        amplitude = frames1
        #print(amplitude)
        f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
        Zxxlog = np.log(np.abs(Zxx)+1e-10)
        
        if lf > 0 :
            lfindex=np.where( f >= lf )
            idx=lfindex[0]
            Zxxlog=Zxxlog[:idx[0],:]
            f=f[:idx[0]]


        if disp :
            
            debugname=self.saveTestPics()  
            
            print('f : ' , f.shape )
            print("t : " , t.shape )
            print("Zxxlog : " , Zxxlog.shape )
            
            fig = plt.figure()
            s = fig.add_subplot(111)
            s.plot(amplitude)
            fig.savefig(debugname+"_wav.png")
            plt.close(fig)
            img0 = cv2.imread(debugname+"_wav.png")
            cv2.imshow("SIGNAL",img0)
            #cv2.waitKey(1)
            
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
            
        
        if save :
            WAVE_OUTPUT_FILENAME=self.savefiles()
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
        
        return Zxxlog

      
    def TestMics_section(self, disp=False , save=False ,recordTime=10 , lowpass=0 ,nperseg=512 , sammeword=False ):  
        pass
        name=self.saveTestFiles()
    
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = int(recordTime/2)  #5s record time
        WAVE_OUTPUT_FILENAME = name
          
        p = pyaudio.PyAudio()
#        print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        frames0 = []
        
#        for i in range(0, int((RATE / CHUNK) * RECORD_SECONDS)):
        frameNumber=int((RATE / CHUNK) * RECORD_SECONDS)
        frameNoise=int((RATE / CHUNK) * 1.5) #3s
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
            
        print("* start recording")
        f=0
        while True:
            data = stream.read(CHUNK)
            data_chunk=array('h',data)
            vol=max(data_chunk)
            if(vol>=Noise):
                print("something said " + str(f))
                frames.append(data)
                snd_start=True
            else:
                pass
                if snd_start : 
                    frames0.append(frames)
                    frames = []
                    snd_start=False
#                print("nothing")
#            print("\n")
            if(f>=frameNumber): break
            f=f+1
        
        print("* done recording")
        
        if(len(frames0)==0) : return
        
#        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        Zxxlog=[]
        ZxxlogN=[]
        

        avg=0
        num=0
        if sammeword : 
            frm_len = []
            for frame_section in frames0 :
                frm_len.append(len(frame_section))
                num=num+1
            avg=max(frm_len)*(0.7)
 

        
        for frame_section in frames0 :
            pass
            ss=0
            frames1 = [] 
            
            if(len(frame_section)<avg): continue
            
            for amp in frame_section :
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
            
            
    #        amplitude = frames1
            #print(amplitude)
            f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
            Zxxlog = np.log(np.abs(Zxx)+1e-10)
            
            if lf > 0 :
                lfindex=np.where( f >= lf )
                idx=lfindex[0]
                Zxxlog=Zxxlog[:idx[0],:]
                f=f[:idx[0]]
    
    
            if disp :
                
                debugname=self.saveTestPics()  
                
                print('f : ' , f.shape )
                print("t : " , t.shape )
                print("Zxxlog : " , Zxxlog.shape )
                
                fig = plt.figure()
                s = fig.add_subplot(111)
                s.plot(amplitude)
                fig.savefig(debugname+"_wav.png")
                plt.close(fig)
                img0 = cv2.imread(debugname+"_wav.png")
                cv2.imshow("SIGNAL",img0)
                #cv2.waitKey(1)
                
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
            
        
            if save :
                WAVE_OUTPUT_FILENAME=self.saveTestFiles()
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frame_section))
                wf.close()
                
            ZxxlogN.append(Zxxlog)
        
        
        
        return ZxxlogN


    def TestMics_whole(self, disp=False , save=False ,recordTime=10 , lowpass=0 ,nperseg=512):  
        pass
        name=self.saveTestFiles()
    
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = int(recordTime/2)  #5s record time
        WAVE_OUTPUT_FILENAME = name
          
        p = pyaudio.PyAudio()
#        print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        frames0 = []
        
#        for i in range(0, int((RATE / CHUNK) * RECORD_SECONDS)):
        frameNumber=int((RATE / CHUNK) * RECORD_SECONDS)
        frameNoise=int((RATE / CHUNK) * 1.5) #3s
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
            
        print("* start recording")
        f=0
        while True:
            data = stream.read(CHUNK)
            data_chunk=array('h',data)
            vol=max(data_chunk)
            if(vol>=Noise):
                print("something said " + str(f))
                frames.append(data)
                snd_start=True
            else:
                pass
                if snd_start : 
                    frames0.append(frames)
                    frames = []
                    snd_start=False
#                print("nothing")
#            print("\n")
            if(f>=frameNumber): break
            f=f+1
        
        print("* done recording")
        
        if(len(frames0)==0) : return
        
#        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        Zxxlog=[]
        
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
            
        #show STFT and WAV
        amplitude=np.zeros([len(frames1)])
        amplitude[:] = frames1
        lf=lowpass
    
        if lf > 0 :
            b, a = signal.butter(10, lf, 'lp', fs=RATE)
#            b, a = signal.iirfilter(10, [lf], btype='low',
#                                   analog=False, ftype='bessel', fs=RATE)
            amplitude = signal.lfilter(b, a, amplitude)
        
        
#        amplitude = frames1
        #print(amplitude)
        f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
        Zxxlog = np.log(np.abs(Zxx)+1e-10)
        
        if lf > 0 :
            lfindex=np.where( f >= lf )
            idx=lfindex[0]
            Zxxlog=Zxxlog[:idx[0],:]
            f=f[:idx[0]]


        if disp :
            
            debugname=self.saveTestPics()  
            
            print('f : ' , f.shape )
            print("t : " , t.shape )
            print("Zxxlog : " , Zxxlog.shape )
            
            fig = plt.figure()
            s = fig.add_subplot(111)
            s.plot(amplitude)
            fig.savefig(debugname+"_wav.png")
            plt.close(fig)
            img0 = cv2.imread(debugname+"_wav.png")
            cv2.imshow("SIGNAL",img0)
            #cv2.waitKey(1)
            
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
        
    
        if save :
            WAVE_OUTPUT_FILENAME=self.saveTestFiles()
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames00))
            wf.close()
    
    
    
        return Zxxlog
    
    
    def signelTestMics_whole(self, recordTime=10 ):  
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
        
#        for i in range(0, int((RATE / CHUNK) * RECORD_SECONDS)):
        frameNumber=int((RATE / CHUNK) * RECORD_SECONDS)
        frameNoise=int((RATE / CHUNK) * 3) #4s
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
            
        print("* start recording")
        f=0
        while True:
            data = stream.read(CHUNK)
            data_chunk=array('h',data)
            vol=max(data_chunk)
            if(vol>=Noise):
                print("something said " + str(f))
                frames.append(data)
                snd_start=True
            else:
                pass
                if snd_start : 
                    frames0.append(frames)
                    frames = []
                    snd_start=False
#                print("nothing")
#            print("\n")
            if(f>=frameNumber): break
            f=f+1
        
        print("* done recording")
        
        if(len(frames0)==0) : 
            print("No Sound detected !")
            return [],0,0,FORMAT,CHANNELS,RATE
        
#        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        
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
        return frames_data,frame_size,dataPerFrame,FORMAT,CHANNELS,RATE
            

        
    def Mics_whole1(self, disp=False , save=True  ,recordTime=10 , lowpass=0 ,nperseg=512, noise=500):  
        pass
        name=self.savefiles()
    
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = int(recordTime/2)  #5s record time
        WAVE_OUTPUT_FILENAME = name
          
        p = pyaudio.PyAudio()
#        print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        frames0 = []
        
#        for i in range(0, int((RATE / CHUNK) * RECORD_SECONDS)):
        frameNumber=int((RATE / CHUNK) * RECORD_SECONDS)
        frameNoise=int((RATE / CHUNK) * 3) #3s
#        frameThreadWait=int((RATE / CHUNK) * 1) #1s
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
            
        print("* start recording")
        
        f=0
#        fd=0
        fr_cnt=0
        while True:
            data = stream.read(CHUNK)
            data_chunk=array('h',data)
            vol=max(data_chunk)
            if(vol>=Noise) and ( fr_cnt<frameNumber ):
                print("something said " + str(f))
                frames.append(data)
                snd_start=True
#                fd=f
                fr_cnt=fr_cnt+1
            else:
                pass
                if snd_start : 
                    frames0.append(frames)
                    frames = []
                    snd_start=False
#                if((f-fd)>frameThreadWait) :
                if ( fr_cnt>=frameNumber ) :
                    fr_cnt=0
                    break
#                    frames0 = []
#                    fr_cnt=0
#                print("nothing")
#            print("\n")
#            if(f>=frameNumber): break
            f=f+1
        
        print("* done recording")
        
        if(len(frames0)==0) : return
        
#        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        Zxxlog=[]
        
        frames00 = []       
        for frame_section in frames0 :
            pass
            frames00.extend(frame_section)
        
        if disp :   
            
            ss=0
            frames1=[]
            for amp in frames00 :
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
            
    
    #        amplitude = frames1
            #print(amplitude)
            f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
            Zxxlog = np.log(np.abs(Zxx)+1e-10)
            
            if lf > 0 :
                lfindex=np.where( f >= lf )
                idx=lfindex[0]
                Zxxlog=Zxxlog[:idx[0],:]
                f=f[:idx[0]]
    
            debugname=self.savePics()
            
            print('f : ' , f.shape )
            print("t : " , t.shape )
            print("Zxxlog : " , Zxxlog.shape )
            
            fig = plt.figure()
            s = fig.add_subplot(111)
            s.plot(amplitude)
            fig.savefig(debugname+"_wav.png")
            plt.close(fig)
            img0 = cv2.imread(debugname+"_wav.png")
            cv2.imshow("SIGNAL",img0)
#                cv2.waitKey(1)
            
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
            
    
        if save : 
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames00))
            wf.close()


    def Mics_whole1Noise(self, disp=False , save=True  ,recordTime=10 , lowpass=0 ,nperseg=512 ,noise=10 ):  
        pass
        name=self.savefiles()
    
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = int(recordTime/2)  #5s record time
        WAVE_OUTPUT_FILENAME = name
          
        p = pyaudio.PyAudio()
#        print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        frames0 = []
        
#        for i in range(0, int((RATE / CHUNK) * RECORD_SECONDS)):
        frameNumber=int((RATE / CHUNK) * RECORD_SECONDS)
        frameNoise=int((RATE / CHUNK) * 3) #3s
#        frameThreadWait=int((RATE / CHUNK) * 1) #1s
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
            
        print("* start recording")
        Noise=noise #hardcode limit
        f=0
#        fd=0
        fr_cnt=0
        while True:
            data = stream.read(CHUNK)
            data_chunk=array('h',data)
            vol=max(data_chunk)
            if(vol<=Noise) and ( fr_cnt<frameNumber ):
                print("something said " + str(f))
                frames.append(data)
                snd_start=True
#                fd=f
                fr_cnt=fr_cnt+1
            else:
                pass
                if snd_start : 
                    frames0.append(frames)
                    frames = []
                    snd_start=False
#                if((f-fd)>frameThreadWait) :
                if ( fr_cnt>=frameNumber ) :
                    fr_cnt=0
                    break
#                    frames0 = []
#                    fr_cnt=0
#                print("nothing")
#            print("\n")
#            if(f>=frameNumber): break
            f=f+1
        
        print("* done recording")
        
        if(len(frames0)==0) : return
        
#        print(len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        Zxxlog=[]
        
        frames00 = []       
        for frame_section in frames0 :
            pass
            frames00.extend(frame_section)
        
        if disp :   
            
            ss=0
            frames1=[]
            for amp in frames00 :
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
            
    
    #        amplitude = frames1
            #print(amplitude)
            f, t, Zxx = signal.stft(amplitude, RATE, nperseg=nperseg)
            Zxxlog = np.log(np.abs(Zxx)+1e-10)
            
            if lf > 0 :
                lfindex=np.where( f >= lf )
                idx=lfindex[0]
                Zxxlog=Zxxlog[:idx[0],:]
                f=f[:idx[0]]
    
            debugname=self.savePics()
            
            print('f : ' , f.shape )
            print("t : " , t.shape )
            print("Zxxlog : " , Zxxlog.shape )
            
            fig = plt.figure()
            s = fig.add_subplot(111)
            s.plot(amplitude)
            fig.savefig(debugname+"_wav.png")
            plt.close(fig)
            img0 = cv2.imread(debugname+"_wav.png")
            cv2.imshow("SIGNAL",img0)
#                cv2.waitKey(1)
            
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
            
    
        if save : 
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames00))
            wf.close()



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
#            thr=threading.Thread(target=self.displaySpectrogramWAV2, args=[name,amplitude,Zxxlog,f,t,Zxx])
#            thr.start()
            self.displaySpectrogramWAV2(name,amplitude,Zxxlog,f,t,Zxx)



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

            
        debugname=self.saveTestPics()    
        
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
        
#        fig = plt.figure(figsize=(14, 8))
        fig = plt.figure()
        plt.pcolormesh(t, f, Zxxlog)
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        #plt.show()
        fig.savefig(debugname+"_stft.png")
        plt.close(fig)
        img = cv2.imread(debugname+"_stft.png")
        cv2.imshow("STFT",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
            
    def destroydisplay(self):
        cv2.destroyAllWindows()
    
    
    def displayResult(self ,
                      disp=True ,
                      pics="IDSS.png",
                      final_label=0 ,
                      final_plabel=0 ,
                      final_owner="" ,
                      final_sound="" ,
                      message="",
                      col=(0,0,255)
                      ):
        
        if disp :
            now = datetime.datetime.now()
            date=now.strftime("%Y/%m/%d")
            tm=now.strftime("%H:%M:%S")
            
            debugname=".\\Pics\\"+pics   
            img = cv2.imread(debugname)

            cv2.putText(img, str(final_owner), (250, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2) 
            cv2.putText(img, str(final_sound) , (250, 100), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2)
            cv2.putText(img, str(final_plabel), (250, 150), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2) 
            cv2.putText(img, str(date)+" "+str(tm), (250, 200), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2) 
            
            cv2.putText(img, str(message), (250, 400), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2) 
            
            cv2.imshow("Panel : Deep Learning and Machine Learning ",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
            
    def displayResult1(self ,
                      disp=True ,
                      pics="IDSS.png",
                      final_label=0 ,
                      final_plabel=0 ,
                      final_owner="" ,
                      final_sound="" ,
                      message="",
                      col=(0,0,255)
                      ):
        
        if disp :
            now = datetime.datetime.now()
            date=now.strftime("%Y/%m/%d")
            tm=now.strftime("%H:%M:%S")
            
            debugname=".\\Pics\\"+pics   
            img = cv2.imread(debugname)

            cv2.putText(img, str(final_owner), (250, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2) 
            cv2.putText(img, str(final_sound) , (250, 100), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2)
            cv2.putText(img, str(final_plabel), (250, 150), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2) 
            cv2.putText(img, str(date)+" "+str(tm), (250, 200), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2) 
            
            cv2.putText(img, "Final Result : ", (5, 248), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,0), 2) 
            cv2.rectangle(img,(5,250),(5+590,250+145),(0,255,0),2)
            cv2.putText(img, str(message), (30, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2) 
            
            cv2.imshow("Panel : Deep Learning and Machine Learning ",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
    

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
    

    def spectrogramWAV1compress(self , disp=False , output=False , lowpass=0 , nperseg=512 , size=0 , frames_data=[] , frame_size=0 , dataPerFrame=0 , FORMAT=pyaudio.paInt16, CHANNELS=2 ,RATE=44100, compressed=0 , logScl=True ):

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
            print(amplitude.shape)
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
            
            debugname=self.saveTestPics()  
            
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
            #cv2.waitKey(1)
            
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
            
#        print("spectrogramWAV1::frames_data " + str(len(frames_data)))
        return Zxxlog
    

    def spectrogramWAV2compress(self , disp=False , output=False , lowpass=0 , name="" , nperseg=512 , size=0 , frames_data=[] , frame_size=0 , dataPerFrame=0 , FORMAT=pyaudio.paInt16, CHANNELS=2 ,RATE=44100, compressed=0 , logScl=True ):

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
            pass
#            thr=threading.Thread(target=self.displaySpectrogramWAV2, args=[name,amplitude,Zxxlog,f,t,Zxx])
#            thr.start()
            self.displaySpectrogramWAV2(name,amplitude,Zxxlog,f,t,Zxx)



#        if disp :   
#    #            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
        return Zxxlog


#record=RecoedSound_Class("KONG","HELLO")  
#for i in range(2):
#    print("Record No."+str(i))
#    record.Mics()#(disp=True)
##print(record.__files)  # private in class
##print(record.names) # obj var of class

#    
#rd=[]   
#rd.append(RecoedSound_Class("KONG","HELLO"))
#rd.append(RecoedSound_Class("OUM","HELLO"))
#rd.append(RecoedSound_Class("KONG","HAY"))
#rd.append(RecoedSound_Class("OUM","HAY"))
#
#for r in rd :
#    for i in range(2):
#        print("Record No. "+str(r.owners)+" , say : "+str(r.sounds))
#        r.Mics()#(disp=True)
