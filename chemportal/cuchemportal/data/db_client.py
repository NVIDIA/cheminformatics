from os import pipe

import sqlalchemy
from cuchemportal.pipeline.pipeline import Pipeline
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import Session, sessionmaker
from typing import Any, Union, Optional
import logging

class DBClient:
    """
    Client class representing a client with 
    CRUD capabilities for cuchemportal pipelines
    database
    """
    def __init__(self, connection_string: Optional[str] = None, 
                connection_config: Optional[dict] = None, pool_count: Optional[int] = 20):
        """
        Builds a DataBase client and connects to database

        Arguments:
            connection
        """
        # Allowing option to simply pass SqlAlchemy connection string
        if connection_string is not None:
            self.engine = create_engine(connection_string)
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
            self.engine = create_engine(self.connection_str)

        # Building and instantiating session object
        self.Session = sessionmaker(bind=self.engine)

        self.metadata = MetaData(bind=self.engine)

        # Displaying basic metadata
        logging.info("Database accessed")
        logging.debug(self.engine.table_names)


    def insert_all(self, *entries: Any) -> bool:
        """Inserts a list of entries into table"""

        # Adding all entries to session and comitting them
        self.session.add_all(entries)
        self.session.commit()
        # Todo: Return array of all inserted objects and make a singular version of this method
        return entries

    def insert(self, record, session: Session) -> bool:
        """Inserts a pipeline entry  into table"""

        # Adding all entries to session and comitting them

        session.add(record)

        # Todo: Return array of all inserted objects and make a singular version of this method
        return record

    def query_id(self, id: int, object_class: Any, session: Session) -> str:
        """Obtains all instances in table of each of a list of queries """

        # Returning first matching pipeline
        query_result = session.query(object_class.id == id).first()
        
        return query_result

    def query_range(self, object_class: Any, start_idx: int, end_idx: int, session: Session) -> str:
        """Obtains first instance in table of each of a list of queries """
         # Returning all pipelines in [start,end)
        query_result = session.query(object_class.id >= start_idx and object_class.id < end_idx).all()
        
        return query_result

    # TODO: fix update methods
    def update_all(self, items_to_change: Any, attribute_to_update: Any, new_value: Any) -> bool:
        """Updates all given database items corresponding to a query""" 
        # Updating all results which are returned by the query
        self.session.query(items_to_change).all().update({attribute_to_update: new_value})
        return True

    def update_record(self, item_to_change: Any, attribute_to_update: Any, new_value: Any) -> bool:
        """Updates the first database item corresponding to a query""" 
        # Updating the first results which is returned by the query
        updated = self.session.query(item_to_change).first().update({attribute_to_update: new_value})
        return updated

    def delete_pipeline(self, pipeline_id: int) -> bool:
        """Deletes every item that matches all queries in a list of queries """
        # Obtaining all corresponding values, deleting and committing
        table = self.metadata.tables["pipelines"]
        table.delete(pipeline_id)

        # boolean validation
        return True 