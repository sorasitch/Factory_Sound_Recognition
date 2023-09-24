# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:40:05 2020

@author: choms
"""
import gspread
from google.oauth2.service_account import Credentials
import datetime

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
import sys
import threading
import time

import cv2


share_flag="NA"
share_int=0

class CloudIOT_Class():
    

    
    def __init__(self,
                 owners="",
                 sounds=""
                 ):
        self.folder_path=".\\SNDPATH"
        self.owners=owners
        self.sounds=sounds
        
        self.scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        self.credentials = Credentials.from_service_account_file(
            'service_account.json',
            scopes=self.scopes
        )
        self.gc = gspread.authorize(self.credentials)
        self.sh = []
        self.worksheet_log = [] 
        self.worksheet_monitor = []
        
        #self.sh = self.gc.open("WorkSheet1")
#        self.sh = self.gc.open_by_url("https://docs.google.com/spreadsheets/d/1_3NI7cKB8WotuU7qrYE1zMJUDmPST1FQFekPpcrhqs4/edit?usp=sharing")


    def googleSheet_init(self,
                         url = "https://docs.google.com/spreadsheets/d/1Xfhd-XMCfgFXWtm1YNDWT-2T_qRC90O366qobs_BBV4/edit?usp=sharing"
                         #url="https://docs.google.com/spreadsheets/d/1_3NI7cKB8WotuU7qrYE1zMJUDmPST1FQFekPpcrhqs4/edit?usp=sharing"
                         ) :
        self.sh = self.gc.open_by_url(url)
        
        
    def googleSheet_updateLog(self,
                              owners="",
                              sounds="",
                              message=""
                              ):
        now = datetime.datetime.now()
        date=now.strftime("%Y/%m/%d")
        tm=now.strftime("%H:%M:%S")
#        print (date)
#        print (tm)
        
        #(1,1)=(1,A)=(row,col)
        self.worksheet_monitor = self.sh.worksheet("LATEST_MONITOR")
        self.worksheet_monitor.update_cell(1, 1, str(date))
        self.worksheet_monitor.update_cell(1, 2, str(tm))
        self.worksheet_monitor.update_cell(1, 3, owners)
        self.worksheet_monitor.update_cell(1, 4, sounds)
        self.worksheet_monitor.update_cell(2, 1, message)
        
        self.worksheet_log = self.sh.worksheet("LOG")
        row=1
        while True:
            val = self.worksheet_log.cell(row, 1).value #(1,1)=(1,A)=(row,col)
            if val=="" :
                self.worksheet_log.update_cell(row, 1, str(date))
                self.worksheet_log.update_cell(row, 2, str(tm))
                self.worksheet_log.update_cell(row, 3, owners)
                self.worksheet_log.update_cell(row, 4, sounds)
                break
                
            row=row+1
            
            
    def googleSheet_updateLog1(self,
                              owners="",
                              sounds="",
                              message=""
                              ):
        now = datetime.datetime.now()
        date=now.strftime("%Y/%m/%d")
        tm=now.strftime("%H:%M:%S")
#        print (date)
#        print (tm)
        
        #(1,1)=(1,A)=(row,col)
        self.worksheet_monitor = self.sh.worksheet("LATEST_MONITOR")
        self.worksheet_monitor.update_cell(1, 1, str(date))
        self.worksheet_monitor.update_cell(1, 2, str(tm))
        self.worksheet_monitor.update_cell(1, 3, owners)
        self.worksheet_monitor.update_cell(1, 4, sounds)
        self.worksheet_monitor.update_cell(2, 1, message)
        
        self.worksheet_log = self.sh.worksheet("LOG")
        self.worksheet_log.insert_row([date, tm, owners, sounds], 2)
#        row=1
#        while True:
#            val = self.worksheet_log.cell(row, 1).value #(1,1)=(1,A)=(row,col)
#            if val=="" :
#                self.worksheet_log.update_cell(row, 1, str(date))
#                self.worksheet_log.update_cell(row, 2, str(tm))
#                self.worksheet_log.update_cell(row, 3, owners)
#                self.worksheet_log.update_cell(row, 4, sounds)
#                break
#                
#            row=row+1
            
            
    def googleMail(self,
                    mail_content='https://docs.google.com/spreadsheets/d/e/2PACX-1vSJUAziUeYPu-bQuH0adeicAC1XNYyfhG4Ciz5ndKej8l464Kx3u9RCC3voDs0yL4_tZK3hB6uwVFWy/pubhtml \n \n https://docs.google.com/forms/d/e/1FAIpQLSeJpA7hEjk2oNb3mzRTEvdFf0Qhtkk4OuD6Z0ESYslIRur_mA/viewform',
                    #mail_content='https://docs.google.com/spreadsheets/d/e/2PACX-1vQv7ngpfo1CdqMihmgqT4BBNHbm07f0tnkZboZ5JEhwWOd0wrgNk3zw3yRYurn9mRWqkzG_sHfF5wsL/pubhtml \n \n https://docs.google.com/forms/d/e/1FAIpQLSd947tm-GuUdk2w-KaoaobunP1FyMhUB7FB9igiD3VULEQvfQ/viewform',
                    sender_address = '@gmail.com', 
                    sender_pass = '123',
                    receiver_address =  '@hotmail.com,@hotmail.com', 
                    subject='pls attention'
                    ):
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        message['Subject'] = subject   #The subject line
        #The body and the attachments for the mail
        message.attach(MIMEText(mail_content, 'plain'))
        #Create SMTP session for sending the mail
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(sender_address, sender_pass) #login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()
        print('Mail Sent')
                                        

    def LINEnotify(self,
                    mail_content='https://docs.google.com/spreadsheets/d/e/2PACX-1vSJUAziUeYPu-bQuH0adeicAC1XNYyfhG4Ciz5ndKej8l464Kx3u9RCC3voDs0yL4_tZK3hB6uwVFWy/pubhtml \n \n https://docs.google.com/forms/d/e/1FAIpQLSeJpA7hEjk2oNb3mzRTEvdFf0Qhtkk4OuD6Z0ESYslIRur_mA/viewform',
                    #mail_content='https://docs.google.com/spreadsheets/d/e/2PACX-1vQv7ngpfo1CdqMihmgqT4BBNHbm07f0tnkZboZ5JEhwWOd0wrgNk3zw3yRYurn9mRWqkzG_sHfF5wsL/pubhtml \n \n https://docs.google.com/forms/d/e/1FAIpQLSd947tm-GuUdk2w-KaoaobunP1FyMhUB7FB9igiD3VULEQvfQ/viewform',
                    token = 'CNS82A5DYuJyl8WS4XrVOEyUyg6efJGJWzdvV9I0mXS',  #'7GhPprCUbHLezSerySs5NG18LiO2Sl1fMD9GkaGKiN0',
                    subject='pls attention'
                    ):
        url = 'https://notify-api.line.me/api/notify'
        headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+token}
        
        msg = subject + ", link : " + mail_content 
        r = requests.post(url, headers=headers, data = {'message':msg})
        print( r.text )


    def googleSheet_responseServer(self
                              ):
        
        for j in range(3) :
            self.displayResult(
                          disp=True ,
                          pics="IDSS2.png",
                          message="",
                          col=(255,0,255)
                          )
        
        global share_flag
        global share_int
#        now = datetime.datetime.now()
#        date=now.strftime("%Y-%m-%d")
#        time=now.strftime("%H:%M:%S")
#        print (date)
#        print (time)
        
        #(1,1)=(1,A)=(row,col)
        worksheet_response = self.sh.worksheet("Form Responses 1")
        row=2
        while True:
            val = worksheet_response.cell(row, 1).value #(1,1)=(1,A)=(row,col)
#            print(worksheet_response.cell(row, 2).value)
            if val=="" :
                break
            else :
                worksheet_response.delete_row(2)
                
#        time.sleep(2)
#        print(share_flag)
        
        while True:
            
            try :
            
                worksheet_response.delete_row(2)
    #            row=1
    #            while True:
    #                val = worksheet_response.cell(row, 1).value #(1,1)=(1,A)=(row,col)
    #                print(worksheet_response.cell(row, 2).value)
    #                if val=="" :
    #                    break
    #                    
    #                row=row+1
    #    
    #            saverow=row
                saverow=2
                val1=""
                while True:
                    val1 = worksheet_response.cell(saverow, 1).value
                    if val1=="" :
                        time.sleep(2)
    #                    continue
                    else :
    #                    print(val1)
                        break
                    
                val2 = worksheet_response.cell(saverow, 2).value
                print(val2)
                if val2.find("SHUTDOWN")>=0 :
                    share_flag=val2
                    share_int=1
                    print("****"+share_flag+"****")
                    self.googleSheet_updateLog1(share_flag,str(share_int),"Control Access")
                    for j in range(3) :
                        self.displayResult(
                              disp=True ,
                              pics="IDSS2.png",
                              message=share_flag,
                              col=(0,0,255)
                              )
                    break
                    sys.exit()
                    
                if val2.find("START")>=0 :
                    share_flag=val2
                    share_int=2
                    print("****"+share_flag+"****")
                    self.googleSheet_updateLog1(share_flag,str(share_int),"Control Access")
                    for j in range(3) :
                        self.displayResult(
                              disp=True ,
                              pics="IDSS2.png",
                              message=share_flag,
                              col=(0,255,0)
                              )
                    
                if val2.find("STOP")>=0 :
                    share_flag=val2
                    share_int=3
                    print("****"+share_flag+"****")
                    self.googleSheet_updateLog1(share_flag,str(share_int),"Control Access")
                    for j in range(3) :
                        self.displayResult(
                              disp=True ,
                              pics="IDSS2.png",
                              message=share_flag,
                              col=(0,0,0)
                              )
    
                if val2.find("ON")>=0 :
                    share_flag=val2
                    share_int=4
                    print("****"+share_flag+"****")
                    self.googleSheet_updateLog1(share_flag,str(share_int),"Control Access")
                    for j in range(3) :
                        self.displayResult(
                              disp=True ,
                              pics="IDSS2.png",
                              message=share_flag,
                              col=(255,255,0)
                              )
    
                if val2.find("OFF")>=0 :
                    share_flag=val2
                    share_int=5
                    print("****"+share_flag+"****")
                    self.googleSheet_updateLog1(share_flag,str(share_int),"Control Access")
                    for j in range(3) :
                        self.displayResult(
                              disp=True ,
                              pics="IDSS2.png",
                              message=share_flag,
                              col=(255,0,255)
                              )
                
            except:
                pass
                time.sleep(2)
            
        self.destroydisplay()    
        print("****googleSheet_responseServer THREAD EXIT****")
#        print("****************************************************THREAD EXIT****")
        
            
    def googleSheet_responseServerThread(self
                          ):     
        thr=threading.Thread(target=self.googleSheet_responseServer, args=[])
        thr.start()
        
    def share_flag_set(self,
                       txt="NA"
                          ): 
        global share_flag
        share_flag=txt
        
    def share_flag_get(self
                          ): 
        global share_flag
        return share_flag
    
    def share_int_set(self,
                       i=0
                          ): 
        global share_int
        share_int=i
        
    def share_int_get(self
                          ): 
        global share_int
        return share_int
        
    def displayResult(self ,
                      disp=True ,
                      pics="IDSS2.png",
                      message="",
                      col=(0,255,0)
                      ):
        
        if disp :
            
            
            now = datetime.datetime.now()
            date=now.strftime("%Y/%m/%d")
            tm=now.strftime("%H:%M:%S")
            
            debugname=".\\Pics\\"+pics   
            img = cv2.imread(debugname)

            cv2.putText(img, "Cloud Control : ", (70, 198), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,0), 2) 
            cv2.rectangle(img,(70,200),(70+500,200+100),(0,255,0),2)   
            cv2.putText(img, str(message), (120, 250), cv2.FONT_HERSHEY_DUPLEX , 0.8, col, 2) 
            
            
            cv2.putText(img, str(date)+" "+str(tm), (350, 50), cv2.FONT_HERSHEY_DUPLEX , 0.6, (0,0,255), 2) 
            
            cv2.imshow("Control Aceess",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
            
    def destroydisplay(self):
        cv2.destroyAllWindows()     
            
#ct = CloudIOT_Class()    

#cldiot = CloudIOT_Class() 
#cldiot.googleSheet_init()
#cldiot.googleSheet_responseServerThread()
#print("exit")
#print(share_flag)
#cldiot.googleSheet_updateLog("555","666","777")
#cldiot.googleMail(subject='pls attention : MachineNumber1')
#cldiot.LINEnotify(subject='pls attention : MachineNumber1')
#cldiot.googleSheet_updateLog1("555","666","777")
#cldiot.googleSheet_updateLog1("888","999","111")
#cldiot.googleSheet_updateLog1("222","333","444")














