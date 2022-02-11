import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger("db_logger")

class DBClient:
    """
    Client class representing a client with
    CRUD capabilities for cuchemportal pipelines
    database
    """
    def __init__(self, connection_str: str, pool_count: Optional[int] = 20):
        """
        Builds a DataBase client and connects to database

        Arguments:
            connection
        """
        # Creating engine, pooling so as to manage connections
        self.engine = create_engine(connection_str,
                                    pool_size=pool_count,
                                    pool_recycle=3600)

        # Building and instantiating session object
        self.Session = sessionmaker(bind=self.engine)

        self.metadata = MetaData(bind=self.engine)

        # Displaying basic metadata
        logger.info("Database connected")
        logger.debug(self.engine.table_names())

    def insert(self, record, session: Session):
        """Inserts a pipeline entry  into table"""
        # Adding record to session
        session.add(record)
        session.flush()
        return record.id

    def query_by_id(self, id: int, db_table: Any, session: Session):
        """Obtains all instances in table of each of a list of queries """

        # Returning first matching pipeline
        query_result = session.query(db_table).filter(db_table.id == id).first()

        return query_result

    def query_range(self, db_table: Any, start_idx: int, n_rows: int, session: Session):
        """Obtains first instance in table of each of a list of queries """
         # Returning all pipelines in [start,end)
        return session.query(db_table).filter(db_table.is_deleted == 0).offset(start_idx).limit(n_rows).all()

    def update(self, db_table: Any, record, session: Session):
        """Updates all given database items corresponding to a query"""
        # Obtaining first value of exact same id (unique so should be only value)
        updatable = session.query(db_table).filter(db_table.id == record.id).first()

        setattr(record, "updated_at", datetime.now())
        for attribute in updatable.__table__.columns:
            # Updating only known attributes on object and mapping back to db
            if hasattr(updatable, attribute.name):
                setattr(updatable, attribute.name, getattr(record, attribute.name))
        # setattr(updatable, "updated_at", datetime.now())

        return record

    def delete(self, db_table: Any, id: int, session: Session):
        """Deletes every item that matches all queries in a list of queries """
        # Obtaining all corresponding values, deleting and committing
        item = session.query(db_table).filter(db_table.id == id).first()
        setattr(item, "is_deleted", 1)
        setattr(item, "updated_at", datetime.now())
        return item