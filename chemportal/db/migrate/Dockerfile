FROM python:3.7-buster

COPY requirements.txt /tmp/
RUN python3 -m pip install -r /tmp/requirements.txt

COPY migrate.py /usr/local/bin
RUN chmod +x /usr/local/bin/migrate.py
