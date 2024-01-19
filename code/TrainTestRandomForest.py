import logging
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
from joblib import dump



class TrainTestRandomForest(object):
    '''
    Purpose
    -------
    This class trains and tests a Random Forest Regression model to estimate wind speed model bias
    '''

    def __init__(self, config, data):
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
        self.data = data
        self.log = logging.getLogger('{}.{}'.format(
            self.config['opts']['projectName'], __name__))

    
    def begin(self):

        X_train, X_val, X_test, y_train, y_val, y_test = self.prepareData()
        if self.config['opts']['trainMLModel']:
            self.trainModel(X_train, X_val, y_train, y_val)
        if self.config['opts']['testMLModel']:
            if self.config['opts']['trainMLModel']==False:
                # If not training model, read in saved model
                from joblib import load
                self.regressor = load(str(self.config['paths']['biasCorrModel']+'rf.joblib'))
            self.testModel(X_test, y_test)

    def prepareData(self):

        self.log.info("  Preparing data")
        self.data=self.data.dropna()

        if self.config['opts']['compareToBaseline']:
            # read data from 1-year database to get exact times used
            self.log.info('Matching times from previous 1-year dataset')
            self.data['keep']=np.nan
            import sqlite3
            connect2=sqlite3.connect(str(self.config['paths']['previous_db']+'db.sqlite'))
            curs2=connect2.cursor()
            for model in ['00z-WRF-GFS','12z-WRF-GFS','00z-WRF-NAM','06z-WRF-NAM','12z-WRF-NAM','18z-WRF-NAM']:
                for loc in ['TANA','PSWP','MAWP','GVWP','BBWF','SFWP','STWP']:
                    self.log.info('Merging times for {} at {}'.format(model,loc))
                    sql="SELECT modelName, initTime, validtime, location, observedWindSpeed FROM data WHERE modelName = '{}' AND location = '{}'".format(model,loc)
                    d=pd.read_sql(sql,connect2)
                    d=d.dropna()
                    d=d.drop(['observedWindSpeed'],axis=1)
                    d['keep']=np.zeros(len(d))+1
                    d['validTime']=pd.to_datetime(d['validTime'].values,format='%Y-%m-%dT%H:%M:%S')
                    d['validTime']=d['validTime'].dt.tz_localize('utc').dt.tz_convert('US/Mountain')
                    d['initTime']=pd.to_datetime(d['initTime'].values,format='%Y-%m-%dT%H:%M:%S')
                    self.data=self.data.merge(d,on=['modelName','initTime','validTime','location'],how='left')
                    self.data['keep_y']=self.data['keep_y'].fillna(self.data['keep_x'])
                    self.data=self.data.drop(['keep_x'],axis=1)
                    self.data=self.data.rename(columns={'keep_y':'keep'})
            self.data=self.data[self.data.keep==1]
            self.data=self.data.drop(['keep','initTime'],axis=1)

        timeOfDay=[t.hour for t in self.data['validTime']]
        self.data['timeOfDay']=timeOfDay
        modelBias=np.array(self.data['windSpeed'].values)-np.array(self.data['ObsWindSpeed'].values)
        self.data['modelBias']=modelBias
        self.data=self.data.drop(['validTime','ObsWindSpeed'],axis=1)
        # Create dummy variables for categorical data
        dummy_model=pd.DataFrame(
            {'modelName':['00z-WRF-GFS','06z-WRF-GFS','12z-WRF-GFS','18z-WRF-GFS','00z-WRF-NAM','06z-WRF-NAM','12z-WRF-NAM','18z-WRF-NAM'],
            'model':[1,2,3,4,5,6,7,8]})
        dummy_loc=pd.DataFrame(
            {'location':['STWP','MAWP','TANA','EHWP','RKWP','HGWP','CAWP','HMWD','BBWF','SFWP','MDWP','GVWP','PSWP','LMWP'],
            'loc':[1,2,3,4,5,6,7,8,9,10,11,12,13,14]})
        self.data=self.data.merge(dummy_model,on=['modelName'],how='left').drop(['modelName'],axis=1)
        self.data=self.data.merge(dummy_loc,on=['location'],how='left').drop(['location'],axis=1)
        if self.config['opts']['compareToBaseline']:
            # Only keep locations that were used in initial analysis
            self.data=self.data[self.data['loc'].isin([1,2,3,9,10,12,13])]
        self.data=self.data.iloc[:,[5,3,0,6,1,2,4]]

        if self.config['opts']['compareToBaseline']:
            if self.config['opts']['testFor1Model'] > 0:
                self.data=self.data[self.data['model']==self.config['opts']['testFor1Model']]
            # We only need 1 set of data here. training and validation dataset are null
            X_train,X_val,y_train,y_val=0,0,0,0
            X_test=self.data.iloc[:,:-1]
            y_test=np.ravel(self.data.iloc[:,-1:],order='C')

        else:
            # Define input/output
            X=self.data.iloc[:,:-1]
            y=np.ravel(self.data.iloc[:,-1:],order='C')
            # Split data into training/validation/testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
            stdev_current=np.std(X_train['windSpeed'].values)
            self.log.info("WINDSPEED STANDARD DEVIATION: {}".format(stdev_current))

        # normalize wind speed by standard deviation
        stdev=float(self.config['parameters']['ws_stdev'])
        if self.config['opts']['compareToBaseline']:
            X_test['windSpeed']=X_test['windSpeed'].values/stdev
        else:
          X_train['windSpeed'], X_val['windSpeed'], X_test['windSpeed'] = X_train['windSpeed'].values/stdev, X_val['windSpeed'].values/stdev, X_test['windSpeed'].values/stdev

        return X_train, X_val, X_test, y_train, y_val, y_test

    def trainModel(self, X_train, X_val, y_train, y_val):

        if self.config['opts']['testHyperparameters']:
            self.log.info("  testing hyperparameters")
            # Define and test algorithms for Random Forest
            # -- n_estimators
            rmse=[]
            for n in list(list(np.arange(5,105,5))+[125,150,200]):
                regressor = RandomForestRegressor(n_estimators=n, random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_val)
                self.log.debug("n_estimators: {} | MAE: {} | RMSE: {}".format(n,metrics.mean_absolute_error(y_val,y_pred),np.sqrt(metrics.mean_squared_error(y_val,y_pred))))
                rmse.append([n,np.sqrt(metrics.mean_squared_error(y_val,y_pred))])
            val=[i for i, j in enumerate(list(list(zip(*rmse))[1])) if j==min(list(list(zip(*rmse))[1]))]
            n_estimators=rmse[val[0]][0]
            self.plot_hyperparamters(list(list(np.arange(5,105,5))+[125,150,200]),list(list(zip(*rmse))[1]),'n_estimators')
            self.log.debug('---- best n_estimators value: {}'.format(n_estimators))

            # -- max_depth
            rmse=[[None,rmse[val[0]][1]]]
            for n in range(1,31):
                regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=n, random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_val)
                self.log.debug("max_depth: {} | MAE: {} | RMSE: {}".format(n,metrics.mean_absolute_error(y_val,y_pred),np.sqrt(metrics.mean_squared_error(y_val,y_pred))))
                rmse.append([n,np.sqrt(metrics.mean_squared_error(y_val,y_pred))])
            val=[i for i,j in enumerate(list(list(zip(*rmse))[1])) if j==min(list(list(zip(*rmse))[1]))]
            max_depth=rmse[val[0]][0]
            self.plot_hyperparamters(range(0,31),list(list(zip(*rmse))[1]),'max_depth')
            self.log.debug('---- best max_depth value: {}'.format(max_depth))

            # -- min_samples_split
            rmse=[]
            for n in list([2,25]+list(np.arange(50,500,50))+list(np.arange(500,1000,100))+list(np.arange(1000,6000,1000))):
                regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=n, random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_val)
                self.log.debug("min_samples_split: {} | MAE: {} | RMSE: {}".format(n,metrics.mean_absolute_error(y_val,y_pred),np.sqrt(metrics.mean_squared_error(y_val,y_pred))))
                rmse.append([n,np.sqrt(metrics.mean_squared_error(y_val,y_pred))])
            val=[i for i,j in enumerate(list(list(zip(*rmse))[1])) if j==min(list(list(zip(*rmse))[1]))]
            min_samples_split=rmse[val[0]][0]
            self.plot_hyperparamters(list([2,25]+list(np.arange(50,500,50))+list(np.arange(500,1000,100))+list(np.arange(1000,6000,1000))),list(list(zip(*rmse))[1]),'min_samples_split')
            self.log.debug('---- best min_samples_split: {}'.format(min_samples_split))

            # -- max_leaf_nodes
            rmse=[[None,rmse[val[0]][1]]]
            for n in list(list(np.arange(5,100,5))+list(np.arange(100,225,25))):
                regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_leaf_nodes=n, random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_val)
                self.log.debug("max_leaf_nodes: {} | MAE: {} | RMSE: {}".format(n,metrics.mean_absolute_error(y_val,y_pred),np.sqrt(metrics.mean_squared_error(y_val,y_pred))))
                rmse.append([n,np.sqrt(metrics.mean_squared_error(y_val,y_pred))])
            val=[i for i,j in enumerate(list(list(zip(*rmse))[1])) if j==min(list(list(zip(*rmse))[1]))]
            max_leaf_nodes=rmse[val[0]][0]
            self.plot_hyperparamters(list([0]+list(np.arange(5,100,5))+list(np.arange(100,225,25))),list(list(zip(*rmse))[1]),'max_leaf_nodes')
            self.log.debug('---- best max_leaf_nodes: {}'.format(max_leaf_nodes))

            # -- min_samples_leaf
            rmse=[]
            for n in list([1,25]+list(np.arange(50,500,50))+list(np.arange(500,1000,100))+list(np.arange(1000,6000,1000))):
                regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=n, random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_val)
                self.log.debug("min_samples_leaf: {} | MAE: {} | RMSE: {}".format(n,metrics.mean_absolute_error(y_val,y_pred),np.sqrt(metrics.mean_squared_error(y_val,y_pred))))
                rmse.append([n,np.sqrt(metrics.mean_squared_error(y_val,y_pred))])
            val=[i for i,j in enumerate(list(list(zip(*rmse))[1])) if j==min(list(list(zip(*rmse))[1]))]
            min_samples_leaf=rmse[val[0]][0]
            self.plot_hyperparamters(list([1,25]+list(np.arange(50,500,50))+list(np.arange(500,1000,100))+list(np.arange(1000,6000,1000))),list(list(zip(*rmse))[1]),'min_samples_leaf')
            self.log.debug('---- best min_samples_leaf: {}'.format(min_samples_leaf))

            # -- max_samples
            rmse=[[None,rmse[val[0]][1]]]
            for n in list([10,50]+list(np.arange(100,1100,100))):
                regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf, max_samples=n, random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_val)
                self.log.debug("max_samples: {} | MAE: {} | RMSE: {}".format(n,metrics.mean_absolute_error(y_val,y_pred),np.sqrt(metrics.mean_squared_error(y_val,y_pred))))
                rmse.append([n,np.sqrt(metrics.mean_squared_error(y_val,y_pred))])
            val=[i for i,j in enumerate(list(list(zip(*rmse))[1])) if j==min(list(list(zip(*rmse))[1]))]
            max_samples=rmse[val[0]][0]
            self.plot_hyperparamters(list([0]+[10,50]+list(np.arange(100,1100,100))),list(list(zip(*rmse))[1]),'max_samples')
            self.log.debug('---- best max_samples: {}'.format(max_samples))

            # -- max_features
            rmse=[]
            for n in ['sqrt',0.2,0.3,0.4,0.5,0.75,1]:
                regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf, max_samples=max_samples, max_features=n, random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_val)
                self.log.debug("max_features: {} | MAE: {} | RMSE: {}".format(n,metrics.mean_absolute_error(y_val,y_pred),np.sqrt(metrics.mean_squared_error(y_val,y_pred))))
                rmse.append([n,np.sqrt(metrics.mean_squared_error(y_val,y_pred))])
            val=[i for i,j in enumerate(list(list(zip(*rmse))[1])) if j==min(list(list(zip(*rmse))[1]))]
            max_features=rmse[val[0]][0]
            self.plot_hyperparamters([0,0.2,0.3,0.4,0.5,0.75,1],list(list(zip(*rmse))[1]),'max_features')
            self.log.debug('---- best max_features: {}'.format(max_features))
        else:
            n_estimators=self.config['parameters']['n_estimators']
            max_depth=self.config['parameters']['max_depth']
            min_samples_split=self.config['parameters']['min_samples_split']
            max_leaf_nodes=None
            min_samples_leaf=self.config['parameters']['min_samples_leaf']
            max_samples=None
            max_features=self.config['parameters']['max_features']

        # Define final model with optimal features
        self.log.info("Creating model with ideal paramters")
        self.regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf, max_samples=max_samples, max_features=max_features, random_state=0)
        self.regressor.fit(X_train, y_train)
        self.log.info("Saving model")
        dump(self.regressor, str(self.config['paths']['biasCorrModel']))

    def plot_hyperparamters(self, x, y, title):

        plt.plot(x, y)
        plt.ylabel('RMSE')
        plt.xlabel(title)
        plt.savefig('{}.png'.format(title))
        plt.close()

    def testModel(self, X_test, y_test):

        self.log.info("Testing Model")

        y_pred = self.regressor.predict(X_test)
        # Evaluate random forest model by comparing outputted bias to actual bias
        self.log.info("RF Model Evaluation | MAE: {} | RMSE: {}".format(metrics.mean_absolute_error(y_test,y_pred),np.sqrt(metrics.mean_squared_error(y_test,y_pred))))

        # y_test_abs is original MAE of the model windspeed before bias correcting
        y_test_abs=[abs(i) for i in y_test]
        # Calculate bias-corrected windspeed from the model
        model_corrected=np.array(X_test['windSpeed'].values)-np.array(y_pred)
        # Calculate observed windspeed from the bias
        obs_windspeed=np.array(X_test['windSpeed'].values)-np.array(y_test)
        # Enter values into dataframe and only keep where observations are > 1 (for MAPE)
        vals=pd.DataFrame({'y_test_abs':y_test_abs,'obs':obs_windspeed,'model_corrected':model_corrected,'loc':X_test['loc'].values})
        vals=vals[vals.obs>1]
        model_corrected_mape=vals['model_corrected'].values
        obs_windspeed_mape=vals['obs'].values
        y_test_abs_mape=vals['y_test_abs'].values

        # Calculate MAPE of original model windspeed
        original_mape=np.mean(np.array(y_test_abs_mape)/np.array(obs_windspeed_mape)*100)
        self.log.info("Original Avg Bias of model WS: {} and MAPE: {}".format(np.mean(y_test_abs),original_mape))
        # Calculate error stats of bias-corrected model windspeed
        new_bias=model_corrected-obs_windspeed
        new_bias_abs=[abs(i) for i in new_bias]
        num=[abs(obs_windspeed_mape[i]-model_corrected_mape[i]) for i in range(len(obs_windspeed_mape))]
        new_mape=np.mean(np.array(num)/np.array(obs_windspeed_mape))*100
        self.log.info("New Avg Bias of corrected WS: {} and MAPE: {}".format(np.mean(new_bias_abs),new_mape))
        self.log.info("New Model Bias Evaluation | MAE: {} | RMSE: {}".format(metrics.mean_absolute_error(model_corrected,obs_windspeed),np.sqrt(metrics.mean_squared_error(model_corrected,obs_windspeed))))


