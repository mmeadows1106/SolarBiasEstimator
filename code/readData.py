import logging
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from metpy.units import units
import metpy.calc as mpcalc
import matplotlib.pyplot as plt


class readData(object):
    '''
    Purpose
    -------
    This class reads model wind speed data from sqlite database and wind speed observations, then matches model and observations based on valid time
    '''

    def __init__(self, config, connection):
        """
        Purpose
        -------
        Initialize the class

        Parameters
        ----------
        config: dictionary
            includes configuration information

        Attributes
        ----------
        None
        """
        self.config = config
        self.connection = connection
        self.log = logging.getLogger('{}.{}'.format(
            self.config['opts']['projectName'], __name__))


    def begin(self):
        
        self.readObs()
        self.readModel()
        self.readObsSitesKey()
        self.matchTimes()


    def readObs(self):

        if self.config['opts']['applyQC']:
            self.log.info("  Reading and QC'ing observation data")
            self.qualityControl()
        else:
            self.log.info("  Reading QC'ed observation data")
            if self.config['opts']['compareToBaseline']:
                self.obsData=pd.read_csv('./input/PI.csv',parse_dates=['Date_Time'])
                # convert SGM data from mph to m/s
                for s in self.config['opts']['SGMSites']:
                    self.obsData[s]=self.obsData[s]*0.44704 
            else:
                self.obsData=pd.read_csv('./output/obs_data_clean.csv',parse_dates=['Date_Time'])
        for col in self.obsData.columns:
            if col=='Date_Time':
                continue
            null=self.obsData.index[self.obsData[col]<0]
            self.obsData[col].iloc[null]=np.nan

    def qualityControl(self):

        # PI sites
        filen=str(Path.cwd())+'\input'+'/PI.csv'
        self.obsData=pd.read_csv(filen,dtype={'STWP':float,'MAWP':float,'TANA':float,'BBWF':float,'SFWP':float,'GVWP':float,'PSWP':float,'FALS14R17':float,'LIME11R8':float})
        # Other sites
        start_time=datetime.datetime(2021,1,1)
        end_time=datetime.datetime(2023,1,1)
        for n in self.config['opts']['obsSites']:
            f=str(n)+'.csv'
            filen=str(Path.cwd())+'\input'+'/'+f
            data=pd.read_csv(filen,header=7) #,dtype={'wind_speed_set_1':float})

            # check timestamps and convert to hourly (top-of-the-hour) if needed
            validTimes=pd.to_datetime(data['Date_Time'].values,format='%Y-%m-%dT%H:%M:%SZ')
            data.drop(columns=['Date_Time','Station_ID'],axis=1,inplace=True)
            data.rename(columns={'wind_speed_set_1':n},inplace=True)
            data['Date_Time']=validTimes
            times=pd.date_range(start=datetime.datetime(2021,1,1),end=datetime.datetime(2022,12,31,23),freq='H')
            missing_times=[]
            for dt in times:
                ind=data.index[data['Date_Time']==dt]
                if len(ind)==0:
                    missing_times.append(dt)
            # Loop through missing top-of-the-hour timesteps
            for dt in missing_times:
                # Check for measurements +/- 30 minutes 
                dt_before=dt-datetime.timedelta(minutes=30)
                dt_after=dt+datetime.timedelta(minutes=30)
                before,after=[],[]
                for dt_test in pd.date_range(start=dt_before,end=dt_after,freq='min'):
                    if dt_test in data['Date_Time'].values:
                        if (dt_test-dt).total_seconds() < 0:
                            before.append(dt_test)
                        if (dt_test-dt).total_seconds() > 0:
                            after.append(dt_test)
                # If measurements available within 30 minutes before AND after, interpolate closest values
                if len(before) > 0 and len(after) > 0:
                    ind_before=data.index[data['Date_Time']==before[-1]]
                    ind_after=data.index[data['Date_Time']==after[0]]
                    ws_before=float(data[n].iloc[ind_before[0]])
                    ws_after=float(data[n].iloc[ind_after[0]])
                    ws=np.interp([(dt-before[-1]).total_seconds()],[0,(after[0]-before[-1]).total_seconds()],[ws_before,ws_after])
                    new_line=pd.DataFrame({'Date_Time':dt,n:ws})
                    data=pd.concat([data.iloc[:ind_after[0]],new_line,data.iloc[ind_after[0]:]]).reset_index(drop=True)                        
                # Otherwise, top-of-the-hour value is null
                else:
                    ind=data.index[data['Date_Time']==dt-datetime.timedelta(hours=1)]
                    new_line=pd.DataFrame({'Date_Time':dt,n:np.nan},index=[0])
                    data=pd.concat([data.iloc[:ind[0]],new_line,data.iloc[ind[0]:]]).reset_index(drop=True)
            data['validTime_minute']=[i.minute for i in data['Date_Time']]
            data=data[data.validTime_minute==0]
            data.index=range(len(data))
            self.obsData=pd.concat([self.obsData,data.drop(columns=['Date_Time','validTime_minute'],axis=1)],axis=1)
        # Convert SGM data from mph to m/s
        for s in self.config['opts']['SGMSites']:
            self.obsData[s]=self.obsData[s]*0.44704
        self.obsData=self.obsData.round(2)
            
        # Remove values where the same value is present for 3 hours in a row (likely not real measurements)
        for col in self.obsData.columns:
            if 'Date_Time' in col:
                continue
            self.obsData.loc[np.where(self.obsData[col]<0)[0],col]=np.nan
            self.scatterplot(col,self.obsData[col].dropna().values,'before')
            data=pd.DataFrame(self.obsData[col])
            data['flag']=data[col].groupby([data[col].diff().ne(0).cumsum()]).transform('size').ge(3).astype(int)
            ind=data.index.where(data['flag']==1)
            self.obsData.loc[ind.dropna(),col]=np.nan
            self.scatterplot(col,self.obsData[col].dropna().values,'after')

        self.obsData.to_csv('./output/obs_data_clean.csv',index=False)


    def scatterplot(self, col, data, c):

        plt.scatter(range(len(data)),data,s=5)
        plt.savefig('./plots/'+col+'_'+c+'.png')
        plt.close()
 
    def readModel(self):

        self.log.info('  Reading model data')
        sql="SELECT modelName, initTime, validTime, forecastHour, location, modelValue FROM data WHERE quantity = 'u10'"
        data=pd.read_sql(sql, self.connection)
        data.rename(columns={'modelValue':'U'},inplace=True)
        sql="SELECT modelName, initTime, validTime, forecastHour, location, modelValue FROM data WHERE quantity = 'v10'"
        data_add=pd.read_sql(sql, self.connection)
        data_add.rename(columns={'modelValue':'V'},inplace=True)
        self.modelData=data.merge(data_add,on=['modelName','initTime','validTime','forecastHour','location'],how='left')

        # Calculate wind speed and direction from U and V
        self.modelData=self.modelData[(self.modelData['U'] > -100) & (self.modelData['V'] > -100)]
        self.modelData['windSpeed']=np.asarray(mpcalc.wind_speed(self.modelData['U'].values*units('m/s'),self.modelData['V'].values*units('m/s')))
        self.modelData['windDir']=np.asarray(mpcalc.wind_direction(self.modelData['U'].values*units('m/s'),self.modelData['V'].values*units('m/s')))
        if self.config['opts']['compareToBaseline']:
            self.modelData=self.modelData.drop(['U','V'],axis=1)
            self.modelData['initTime']=pd.to_datetime(self.modelData['initTime'].values,format='%Y-%m-%dT%H:%M:%S')
        else:
            self.modelData.drop(['initTime','U','V'],axis=1,inplace=True)
        self.modelData['validTime']=pd.to_datetime(self.modelData['validTime'].values,format='%Y-%m-%dT%H:%M:%S')
        self.modelData['validTime']=self.modelData['validTime'].dt.tz_localize('utc').dt.tz_convert('US/Mountain')
        self.modelData=self.modelData[self.modelData.validTime.dt.year < 2023].reset_index(drop=True)

    def readObsSitesKey(self):

        filen=('./input/obs_sites.csv')
        self.obsSitesKey=pd.read_csv(filen)

    def matchTimes(self):
        
        self.log.info("  Matching observation and model times")

        obsData=self.obsData.copy()
        obsData.rename(columns={'ODT56':'EHWP','FALS14R17':'RKWP','ODT81':'HGWP','ITD14':'CAWP','ITD21':'MDWP','LIME11R8':'LMWP','Date_Time':'validTime'},inplace=True)
        if not self.config['opts']['compareToBaseline']:
            obsData['HMWD']=obsData['CAWP']
        obsData['validTime']=obsData['validTime'].dt.tz_localize('US/Mountain',ambiguous='infer')

        self.data=pd.DataFrame()
        if self.config['opts']['compareToBaseline']:
            locs=['STWP','MAWP','TANA','BBWF','SFWP','GVWP','PSWP']
        else:
            locs=['STWP','MAWP','TANA','EHWP','RKWP','HGWP','CAWP','HMWD','BBWF','SFWP','MDWP','GVWP','PSWP','LMWP']
        for loc in locs:
            df_model=self.modelData[self.modelData.location==loc]
            df_obs=obsData[['validTime',loc]]
            df_obs=df_obs.rename(columns={loc:'ObsWindSpeed'})
            data=df_model.merge(df_obs,on=['validTime'],how='left')
            self.data=pd.concat([self.data,data])
        self.data.reset_index()
