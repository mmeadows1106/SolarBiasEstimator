"""
Driver

Author: Melinda Meadows
Created: 2024-01-10
"""

import logging
import yaml
import os
import datetime
import sys
from pushover import Client
from shutil import copyfile
from pathlib import Path
import sqlite3
from sqlTools import sqlTools
from readData import readData
from TrainTestRandomForest import TrainTestRandomForest
from BiasCorrectHistorical import BiasCorrectHistorical
# /// Functions ///


def init_logger(config):
    """
    Function to initialize a logger and create appdata directories if needed

    Parameters
    ----------
    projectName: string
        name of the project

    Attributes
    ----------
    log_file: string
        path of the log file
    logger: log object
        provides log object
    logger_handler: log file handler
        log file handler
    logger_formatter: log formatter
        provides formatting for log strings


    Returns
    -------
    logger: object
        formatted log object

    """

    projectName = config['opts']['projectName']
    #log_file = os.environ.get('APPDATA')+'\\{}\\logs\\{}.log'.format(projectName, projectName)
    log_file = os.getcwd()+'\\{}.log'.format(projectName)

    if os.path.isfile(log_file):
        try:
            os.remove(log_file)
        except:
            pass
    # Create the Logger
    logger = logging.getLogger(projectName)
    if config['opts']['debugMode']:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Create the Handler for logging data to a file
    logger_handler = logging.FileHandler(log_file)
    if config['opts']['debugMode']:
        logger_handler.setLevel(logging.DEBUG)
    else:
        logger_handler.setLevel(logging.INFO)

    # Create a Formatter for formatting the log messages
    logger_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s:%(lineno)d | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')

    # Add the Formatter to the Handler
    logger_handler.setFormatter(logger_formatter)

    # Add the Handler to the Logger
    logger.addHandler(logger_handler)

    return(logger)


def read_config():
    """
    Read the config file


    Attributes
    ----------
    f: file object
        opened file

    Returns
    -------
    config: dictionary
        configuration dictionary    
    """
    with open(os.getcwd()+'/config/config.yaml') as f:
        return(yaml.load(f, Loader=yaml.FullLoader))


# Main code
if __name__ == "__main__":

    config = read_config()
    # this client is for push notifications to a cell phone
    # it uses this: https://pushover.net/
    client = Client(config['opts']['pushoverUserToken'],
                   api_token=config['opts']['pushoverAppToken'])
    log = init_logger(config)

    log.info('Start')
    log.info('Working Directory: {}'.format(os.getcwd()))
    script_start_time = datetime.datetime.now()

    try:

        ''' class calls here '''

        log.info(" Step 0. Connect to sqlite database")
        connection = sqlite3.connect(config['paths']['db']+'db.sqlite')
        cursor = connection.cursor()
        sTools = sqlTools(config, cursor, connection)

          
        if config['opts']['testMLModel'] or config['opts']['trainMLModel']:
            log.info(" Step 1. Read data")
            d=readData(config, connection)
            d.begin()
            log.info(" Step 2. Applying random forest regression")
            rf=TrainTestRandomForest(config, d.data)
            rf.begin()
        else:
            log.info(" Skipping training and testing steps. Reading in previously-created ML model")

        if config['opts']['biasCorrectHistorical']:
            log.info(" Step 3. Bias correcting historical data")
            h=BiasCorrectHistorical(config)
            h.begin()

        log.info('SUCCESS')
    except:
        log.exception('Problem in driver')
        if not config['opts']['debugMode']:
            copyfile(str(Path.cwd() / '{}.log'.format(config['opts']['projectName'])), str(Path.cwd(
            ) / '{}_{}.log'.format(config['opts']['projectName'], datetime.datetime.now().strftime('%Y%m%d_%H%M'))))
            client.send_message('my Message', title='message title')

    finally:
        end_time = datetime.datetime.now()
        log.info('Total execution time: ' + str(end_time - script_start_time))
        handlers = log.handlers[:]
        for handler in handlers:
            handler.close()
            log.removeHandler(handler)
