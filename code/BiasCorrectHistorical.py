import logging
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from joblib import load
import sqlite3
import metpy.calc as mpcalc
from metpy.units import units


class BiasCorrectHistorical(object):
    '''
    Purpose
    -------
    This class reads in u and v wind data from a given database for a given timeframe, bias corrects the wind speed, then writes the bias-corrected u and v components to the same self.database
    '''

    def __init__(self, config):
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
        self.log = logging.getLogger('{}.{}'.format(
            self.config['opts']['projectName'], __name__))

    def begin(self):

        self.log.info('  Reading and preparing data')
        self.readData()
        self.log.info('  Bias-correcting data')
        self.biasCorrect()
        self.log.info('  Saving data to database')
        self.saveData()


    def readData(self):

        if self.config['opts']['histDB_sqlite']:
            db=self.config['opts']['pathHistoricalDB']+'db.sqlite'
            self.connection = sqlite3.connect(db)
        else:
            from sqlalchemy import create_engine
            self.engine = create_engine(self.config['opts']['pathHistoricalDB'])
            self.connection = self.engine.connect()
            self.connection.execution_options(isolation_level="AUTOCOMMIT")
        self.log.debug('   Connected to database')
               
        starttime=self.config['opts']['startTime']
        endtime=self.config['opts']['endTime']

        sql="SELECT fileID, modelName, initTime, validTime, forecastHour, location, units, modelValue FROM data WHERE quantity = 'u10'"
        u=pd.read_sql(sql, self.connection)
        u.rename(columns={'modelValue':'U'},inplace=True)
        sql="SELECT fileID, modelName, initTime, validTime, forecastHour, location, units, modelValue FROM data WHERE quantity = 'v10'"
        v=pd.read_sql(sql, self.connection)
        v.rename(columns={'modelValue':'V'},inplace=True)
        u=u.drop_duplicates(subset=['fileID','modelName','initTime','validTime','forecastHour','location','units'])
        v=v.drop_duplicates(subset=['fileID','modelName','initTime','validTime','forecastHour','location','units'])
        self.data=u.merge(v,on=['fileID','modelName','initTime','validTime','forecastHour','location','units'],how='inner')
        self.data_copy=self.data.copy()
        self.data.drop(['fileID','units'],axis=1,inplace=True)
        self.log.debug('   Read data')

        self.data['windSpeed']=np.asarray(mpcalc.wind_speed(self.data['U'].values*units('m/s'),self.data['V'].values*units('m/s')))
        self.data['windDir']=np.asarray(mpcalc.wind_direction(self.data['U'].values*units('m/s'),self.data['V'].values*units('m/s')))
        self.data.drop(['U','V'],axis=1,inplace=True)
        self.data['validTime']=pd.to_datetime(self.data['validTime'].values,format='%Y-%m-%dT%H:%M:%S')
        self.data['initTime']=pd.to_datetime(self.data['initTime'].values,format='%Y-%m-%dT%H:%M:%S')
        if starttime != 0 and endtime != 0:
            mask=(self.data['initTime'] >= starttime) & (self.data['initTime'] <= endtime)
            self.data=self.data.loc[mask]
            self.data_copy=self.data_copy[mask]
        self.data['timeOfDay']=[t.hour for t in self.data['validTime']]
        self.data.drop(['validTime','initTime'],axis=1,inplace=True)

        dummy_model=pd.DataFrame(
            {'modelName':['00z-WRF-GFS','06z-WRF-GFS','12z-WRF-GFS','18z-WRF-GFS','00z-WRF-NAM','06z-WRF-NAM','12z-WRF-NAM','18z-WRF-NAM'],
            'model':[1,2,3,4,5,6,7,8]})
        dummy_loc=pd.DataFrame(
            {'location':['STWP','MAWP','TANA','EHWP','RKWP','HGWP','CAWP','HMWD','BBWF','SFWP','MDWP','GVWP','PSWP','LMWP'],
            'loc':[1,2,3,4,5,6,7,8,9,10,11,12,13,14]})
        self.data=self.data.merge(dummy_model,on=['modelName'],how='left').drop(['modelName'],axis=1)
        self.data=self.data.merge(dummy_loc,on=['location'],how='left').drop(['location'],axis=1)
        self.data=self.data.iloc[:,[4,3,0,5,1,2]]
        self.data['windSpeed']=self.data['windSpeed'].values/self.config['parameters']['ws_stdev']
        self.log.debug('   Prepared data')

    def biasCorrect(self):

        regressor=load(self.config['paths']['biasCorrModel'])
        wind_speed_bias=regressor.predict(self.data)
        wind_speed_corrected=np.array(self.data['windSpeed'].values)-np.array(wind_speed_bias)
        wind_components_corrected=mpcalc.wind_components(wind_speed_corrected*units('m/s'),self.data['windDir'].values*units.deg)

        self.data_copy['u_corrected']=np.asarray(wind_components_corrected[0])
        self.data_copy['v_corrected']=np.asarray(wind_components_corrected[1])
        u_corrected=self.data_copy.drop(['U','V','v_corrected'],axis=1).rename(columns={'u_corrected':'modelValue'})
        u_corrected['quantity']=['u10_corrected']*len(u_corrected)
        v_corrected=self.data_copy.drop(['U','V','u_corrected'],axis=1).rename(columns={'v_corrected':'modelValue'})
        v_corrected['quantity']=['v10_corrected']*len(v_corrected)
        self.wind_corrected=pd.concat([u_corrected,v_corrected])
        self.wind_corrected=self.wind_corrected.iloc[:,[0,1,2,3,4,5,8,6,7]]

    def saveData(self):

        self.wind_corrected=self.wind_corrected.sort_values(by=['modelName','initTime','forecastHour']).copy()
        if self.config['opts']['histDB_sqlite']:
            sql="SELECT * FROM data ORDER BY id DESC LIMIT 1"
        else:
            from sqlalchemy import text
            sql=text("SELECT TOP 1 * FROM wind.data ORDER BY id DESC")
        result=pd.read_sql(sql,self.connection)
        if result.shape[0]>0:
            self.wind_corrected.set_index(np.arange(result['id'].values+1,result['id'].values+self.wind_corrected.shape[0]+1),inplace=True,drop=True)
        self.wind_corrected.index.name='id'

        if self.config['opts']['histDB_sqlite']:
            self.wind_corrected.to_sql('data',self.connection,if_exists='append')
        else:
            self.wind_corrected.to_sql('data',self.engine,if_exists='append',schema='wind')

        self.log.debug('   Saved data')