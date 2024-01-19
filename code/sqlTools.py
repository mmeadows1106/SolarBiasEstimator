import logging
import sys


class sqlTools(object):
    def __init__(self, config, cursor, connection):
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
        self.cursor = cursor
        self.connection = connection

    def init_db(self):
        """
        Routine to create an empty data table

        Attributes
        ----------
        sql: string
            sql query string
        """
        self.log.info('Initialize data table')

        self.sql = "CREATE TABLE data (id integer,\
            modelName string,\
            initTime string, \
            validTime string,\
            forecastHour integer,\
            location string,\
            modelWindSpeed float,\
            observedWindSpeed float,\
            observedGeneration float);"
        self.cursor.execute(self.sql)

    def findNextRowID(self, tableName):
        """
        Returns the next row id in a table

        Parameters
        ----------
        tableName: string
            name of the table to check

        Attributes
        ----------
        cursor: object
            cursor for sqlite db
        sql: string
            sql command to execute
        result: int
            last row ID

        Returns
        -------
        result: int
            next row ID
        """
        sql = "SELECT * FROM {} ORDER BY id DESC LIMIT 1".format(tableName)
        self.cursor.execute(sql)
        try:
            result = self.cursor.fetchall()[0][0]
            result += 1
        except:
            result = 0

        return(int(result))
