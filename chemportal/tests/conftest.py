import time
import pytest
import subprocess
import testinfra

DOCKER_IMAGE_NAME = 'cuchem_db_img'


@pytest.fixture(scope='session')
def mysql_server(request):
    # Start MySQL server container
    docker_id = subprocess.check_output(['docker', 'run', '-d', DOCKER_IMAGE_NAME]).decode().strip()
    mysql_server = testinfra.get_host("docker://" + docker_id)
    request.mysql_server = mysql_server
    mysql_server_ip = mysql_server.interface("eth0").addresses[0]

    while not mysql_server.socket(f'tcp://{mysql_server_ip}:3306').is_listening:
        time.sleep(1)

    yield mysql_server

    # DB container cleanup
    subprocess.check_call(['docker', 'rm', '-f', docker_id])
