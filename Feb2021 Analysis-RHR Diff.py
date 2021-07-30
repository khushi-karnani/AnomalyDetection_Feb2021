#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import os


#importing intraday_heartrate data into DataFrames from df1 to df28
folder = './'

filelist = [file for file in os.listdir(folder) if file.startswith("intraday_heartrate_2021-02-")]
i=1
data = []
for file in filelist:
    exec("df%d = pd.read_csv('%s')" % (i, os.path.join(folder,file)))
    exec("df%d['filename'] = '%s'" % (i,os.path.basename(file).split('.')[0].split('_')[2]+" "))
    exec("data.append(df%d)" % (i))
    i=i+1
    
    
#importing intraday_steps_cal_elev data into DataFrames from df1_step to df28_step
folder = './'

filelist = [file for file in os.listdir(folder) if file.startswith("intraday_steps_cal_elev_2021-02-")]
i=1
data = []
for file in filelist:
    exec("df%d_step = pd.read_csv('%s')" % (i, os.path.join(folder,file)))
    exec("df%d_step['filename'] = '%s'" % (i,os.path.basename(file).split('.')[0].split('_')[4]+" "))
    exec("data.append(df%d_step)" % (i))
    i=i+1
    
    
#importing intraday_sleep data into DataFrames from df1_sleep to df28_sleep
folder = './'

filelist = [file for file in os.listdir(folder) if file.startswith("intraday_sleep_2021-02-")]
i=1
data = []
for file in filelist:
    exec("df%d_sleep = pd.read_csv('%s')" % (i, os.path.join(folder,file)))
    exec("df%d_sleep['filename'] = '%s'" % (i,os.path.basename(file).split('.')[0].split('_')[2]+" "))
    exec("data.append(df%d_sleep)" % (i))
    #missing data for day 23 and 26
    if i==22 or i==25:
        i+=2
    else:
        i+=1

        
#preparing the raw heart rate datasets (df1 to df29) for merging

#merge column date(filename) and time
for i in range(1,29):
    exec("df%d['Time'] = df%d['filename'] +" "+ df%d['Time']"%(i,i,i))
    exec("df%d.drop(['filename'],axis=1,inplace=True)"%(i))
    
#convert into dfi['Time'] to timestamp
for i in range(1,29):
    exec("df%d['Time'] = pd.to_datetime(df%d['Time'])"%(i,i)) 
    
#converting datetime64[ns] to DatetimeIndex
for i in range(1,29):
    exec("df%d=df%d.set_index(['Time'])"%(i,i))
    
#Averaging seconds level heart rate for each minute to get minute level heart rate.
for i in range(1,29):
    exec("df%d=df%d['HeartRate'].resample('1Min').mean()"%(i,i))
    exec("df%d=pd.DataFrame(df%d)"%(i,i))
    

#preparing the raw steps data (df1_step to df29_step) for merging

#merge column date(filename) and Time
for i in range(1,29):
    exec("df%d_step['Time'] = df%d_step['filename'] +" "+ df%d_step['Time']"%(i,i,i))
    exec("df%d_step.drop(['filename'],axis=1,inplace=True)"%(i))
    
#convert into dfi['Time'] to datetime64[ns]
for i in range(1,29):
    exec("df%d_step['Time'] = pd.to_datetime(df%d_step['Time'])"%(i,i))

#converting datetime64[ns] to DatetimeIndex
for i in range(1,29):
    exec("df%d_step=df%d_step.set_index(['Time'])"%(i,i))
     

#preparing the raw sleep data (df1_sleep to df29_sleep) for merging

#converting 'Timestamp' column to datetime object
for i in range(1,29):
    if(i==23 or i==26):
        continue
    exec("df%d_sleep['Timestamp'] = pd.to_datetime(df%d_sleep['Timestamp'])"%(i,i))

#converting datetime64[ns] to DatetimeIndex and unifying column name to match the other datasets
for i in range(1,29):
    if(i==23 or i==26):
        continue
    exec("df%d_sleep.rename(columns={'Timestamp': 'Time'},inplace=True)"%(i))
    exec("df%d_sleep=df%d_sleep.set_index(['Time'])"%(i,i))
    
#creating dummy df23_sleep and df26_sleep dataframe  
df23_sleep=df1_sleep
df26_sleep=df1_sleep

a=['Minutes Asleep','Sleep Type','filename']
for i in a:
    df23_sleep[i]=np.nan
    df26_sleep[i]=np.nan
    

    
#merging columns of all datasets to form daywise datasets having all columns (di_allcols for day i)
for i in range(1,29):
    exec("d%d_allcols = pd.merge(df%d,pd.merge(df%d_step,df%d_sleep, on='Time',how='outer'), on='Time',how='outer')"%(i,i,i,i))

#merging all di_allcols into a single dataframe
list_di_allcols=[d2_allcols,d3_allcols,d4_allcols,d5_allcols,d6_allcols,d7_allcols,d8_allcols,d9_allcols,d10_allcols,d11_allcols,d12_allcols,d13_allcols,d14_allcols,d15_allcols,d16_allcols,d17_allcols,d18_allcols,d19_allcols,d20_allcols,d21_allcols,d22_allcols,d23_allcols,d24_allcols,d25_allcols,d26_allcols,d27_allcols,d28_allcols]
df_feb_merged=d1_allcols.append(list_di_allcols) #dataframe having data for the entire month




#RHR 
class RHR:
    
    # calculate and filter resting heart rate
    
    def resting_heart_rate(self, df_feb_merged): #input the dataset containing data for the entire month
        df_feb_data=df_feb_merged
        df_feb_data['Steps_window_12'] = df_feb_data['Steps'].rolling(12).sum()
        df_feb_data = df_feb_data.loc[(df_feb_data['Steps_window_12'] == 0)]
        return df_feb_data #has resting heart rate data only
    
    # preprocessing
    
    def preprocessing(self, df_feb_data): #input dataset containing resting heart rate values
        data=df_feb_data.drop(['Calories Burnt','Minutes Asleep','Sleep Type','Steps_window_12','filename','Elevation (m)'],axis=1)
        # smooth data
        data_nonas = data.dropna()
        data_rom = data_nonas.rolling(400).mean()
        # resample
        data_resmp = data_rom.resample('1H').mean()
        data = data_resmp.drop(['Steps'], axis=1)
        data = data.dropna()
        return data
    
    # grouping and then standardizing the resting heart rate values
    
    def group_normalize(self, data): #input dataset containing preprocessed data
        
        #grouping according to weekdays and weekends
        data_backup=data.reset_index()

        data['dayofweek']=data.index.strftime("%A")
        data['Weekday']=[[''] for c in range(0,598)]
        for i in range(0,598):
            if data['dayofweek'][i]=='Sunday' or data['dayofweek'][i]=='Saturday':
                data['Weekday'][i]='2. No'
            else:
                data['Weekday'][i]='1. Yes'

        data=data.set_index('Weekday')
        data=data.drop(['dayofweek'],axis=1)
        
        #normalizing the resting heart rates by grouping them separately for weekdays and weekends
        df_rescaled = data.groupby(data.index).apply(StandardScaler().fit_transform)

        flat_list = []
        for sublist in df_rescaled:
            for item in sublist:
                flat_list.append(item)

        data_backup['re_scaled']=0.0
        for i in range(0,598):
            data_backup['re_scaled'][i]=flat_list[i][0]

        data_backup=data_backup.set_index('Time')
        data = data_backup.drop(['HeartRate'],axis=1).fillna(0)
        data=data.rename(columns={"re_scaled": "HeartRate"})

        return data #contains the normalized heart rates
    
    
    def anomaly_detection(self, data): #input normalized resting heart rate data
        """
        This function takes the standardized data and detects outliers using Gaussian density estimation.
        """
        model =  EllipticEnvelope(contamination=0.1,random_state=10, support_fraction=0.7)
        model.fit(data)
        preds = pd.DataFrame(model.predict(data))
        preds = preds.rename(lambda x: 'anomaly' if x == 0 else x, axis=1)
        data = data.reset_index()
        data = data.join(preds)
        return data
    
    
    def visualize(self, data): #input dataset having anomaly detection predictions
        plt.rcdefaults()
        fig, ax = plt.subplots(1, figsize=(80,15))
        a = data.loc[data['anomaly'] == -1, ('Time', 'HeartRate')] #anomaly
        b = a[(a['HeartRate'] > 0)]
        ax.bar(data['Time'], data['HeartRate'], linestyle='-',color='midnightblue' ,lw=6, width=0.02)
        ax.scatter(a['Time'],a['HeartRate'], color='red', label='Anomaly', s=500) #used a instead of b

        ax.tick_params(axis='both', which='major', color='blue', labelsize=50)
        ax.tick_params(axis='both', which='minor', color='blue', labelsize=50)
        ax.set_title('February 2021',fontweight="bold", size=70) # Title
        ax.set_ylabel('Std. RHR\n', fontsize = 70) 

        plt.plot()
        plt.savefig('RHR.jpg', dpi=300, bbox_inches='tight')
    
model = RHR()

df_feb_data = model.resting_heart_rate(df_feb_merged)
data = model.preprocessing(df_feb_data)
norm_data = model.group_normalize(data)
data = model.anomaly_detection(norm_data)
model.visualize(data)

#ignore the warnings, they occur because I am overwritting values while grouping

