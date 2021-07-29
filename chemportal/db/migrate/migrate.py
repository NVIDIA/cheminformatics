#!/usr/bin/env python

import sys
import logging
import mysql.connector as mysql

logger = logging.getLogger('pef-db-copy')
console_handle = logging.StreamHandler(sys.stdout)
console_handle.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(name)s [%(levelname)s]: %(message)s')
console_handle.setFormatter(formatter)

logger.addHandler(console_handle)
logger.setLevel(logging.DEBUG)

source_db = mysql.connect(
    host="10.33.254.15",
    user="root",
    passwd="pef@@12345678",
    database="pef_new"
)

target_db = mysql.connect(
    host="sc-gpu-db-pef-write.nvidia.com",
    user="pef_usr",
    passwd="re3mEtPyaQ",
    database="pef"
)


def get_last_id(table_name):
    cursor = target_db.cursor()

    query = "SELECT max(id) FROM " + table_name
    cursor.execute(query)
    record = cursor.fetchone()
    last_id = record[0]
    if last_id is None:
        last_id = -1
    return last_id


def get_new_records(table_name, start_id):
    src_cursor = source_db.cursor()
    query = "SELECT * FROM %s where id > %d" % (table_name, start_id)
    src_cursor.execute(query)
    records = src_cursor.fetchall()
    row_headers = [x[0] for x in src_cursor.description]
    return row_headers, records


def upload_table(table_name):
    id = get_last_id(table_name)
    logger.info('Coping records from table %s with starting id %d...' % (
        table_name, id))
    row_headers, new_records = get_new_records(table_name, id)

    insert_stmt = "INSERT INTO %s ( %s ) VALUES (%s)" % (
        table_name, ', '.join(row_headers),
        (len(row_headers) * '%s, ').strip(', '))

    logger.info("Insert statement generated for table %s is %s" % (
        table_name, insert_stmt))

    target_cursor = target_db.cursor()
    target_cursor.executemany(insert_stmt, new_records)
    target_db.commit()

    logger.info("%d records inserted" % target_cursor.rowcount)


upload_table('request')
upload_table('request_state')
upload_table('task_performance')
upload_table('task_performance_detail')
upload_table('task_performance_versions ')
