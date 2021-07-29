import sqlalchemy as sa
from sqlalchemy import engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import MetaData, Table
import pandas as pd
from typing import Any, Union, Optional
import logger

class DBClient:
    """
    Client class representing a client with 
    CRUD capabilities for cuchemportal pipelines
    database
    """
    def __init__(self, connection_string: Optional[str], 
                connection_config: Optional[dict], pool_count: Optional[int] = 20):
        """
        Builds a DataBase client and connects to database

        Arguments:
            connection
        """
        # Allowing option to simply pass SqlAlchemy connection string
        if connection_string is not None:
            self.engine = sa.create_engine(connection_string, 
                        pool_size = pool_count, max_overflow = 0)
        # Otherwise obtaining values as parameters
        else:
            # Obtaining credentials from config and building connection string
            dbuser = connection_config["dbuser"]
            dbpass = connection_config["dbpass"]
            dbhost = connection_config["dbhost"]
            dbname = connection_config["dbname"]
            self.connection_str = ("mysql+pymysql://" 
                                     "{0}:{1}@{2}"
                                     "{3}".format(dbuser, dbpass, dbhost, dbname))

            # Creating engine, pooling so as to manage connections
            self.engine = sa.create_engine(self.connection_str, 
                        pool_size =  pool_count, max_overflow = 0)

        # Building connection and session objects
        self.connection = self.engine.connect()
        self.session = sessionmaker(bind=engine)

        # Displaying basic metadata
        logger.info("Database accessed")
        logger.debug(self.engine.table_names)


    def insert_all(self, *entries: Any) -> bool:
        """Inserts a list of entries into table"""

        # Adding all entries to session and comitting them
        self.session.add_all(entries)
        self.session.commit()
        # Todo: Return array of all inserted objects and make a singular version of this method
        return entries

    def insert(self, entry: Any) -> bool:
        """Inserts a list of entries into table"""

        # Adding all entries to session and comitting them
        self.session.add(entry)
        self.session.commit()
        # Todo: Return array of all inserted objects and make a singular version of this method
        return entry

    def query_all(self, *queries: Any) -> str:
        """Obtains all instances in table of each of a list of queries """
        query_result = ""

        # Appending all results for every query
        for query in queries:
            query_result += self.session.query(query).all() + "\n"
        
        return query_result

    def query_first(self, *queries: Any) -> str:
        """Obtains first instance in table of each of a list of queries """
        query_result = ""

        # Appending first result from every query
        for query in queries:
            query_result += self.session.query(query).first() + "\n"
        
        return query_result

    # TODO: fix update methods
    def update_all(self, items_to_change: Any, attribute_to_update: Any, new_value: Any) -> bool:
        """Updates all given database items corresponding to a query""" 
        # Updating all results which are returned by the query
        self.session.query(items_to_change).all().update({attribute_to_update: new_value})
        return True

    def update_first(self, item_to_change: Any, attribute_to_update: Any, new_value: Any) -> bool:
        """Updates the first database item corresponding to a query""" 
        # Updating the first results which is returned by the query
        updated = self.session.query(item_to_change).first().update({attribute_to_update: new_value})
        return updated

    def delete(self, *queries: list) -> bool:
        """Deletes every item that matches all queries in a list of queries """
        for query in queries:
            # Obtaining all corresponding values, deleting and committing
            to_delete = self.session.query(query).all()
            self.session.delete(to_delete)
            self.session.commit(to_delete)

        # boolean validation
        return True 