#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import http.client
import json
import ssl
import getpass
import sys

from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Colossus():
    """
    Implements all tasks to acquire and manage machines from Colossus.
    """

    server = 'colossus.nvidia.com'
    authentication_token = None

    def _fetch_machine_schedules(self,
                                 gpu_type='RTX-A6000',
                                 poolname='app-validation'):
        """
        Retrieves machine schedules from Colossus with custom filters around
        GPU type and OS.
        """
        if Colossus.authentication_token is None:
            raise Exception("Call login before trying to lease a machine.")
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=1)

        input_data = {"startTime": start_date.isoformat(),
                      "endTime": end_date.isoformat(),
                      "gpuTypeList": [gpu_type],
                      "poolNameList": ['general', poolname]}

        headers = {'authorization':
                   'Bearer ' + Colossus.authentication_token['authToken']}

        status, machine_list = self.post(Colossus.server,
                                    "/v3/resource/filter-lease",
                                    input_data, headers)

        logger.info("Fetching machines returned %d" % status)

        return machine_list

    def _idle_machine(self, machines_schedule, start_date, end_date):
        """
        Fetch idle machines from Colossus machining a start-end dates and
        GPU types, suitable for leasing.
        """
        available_machines = list(filter(
            lambda m: m['resourceStatus'] == 'AVAILABLE',
            machines_schedule['resourceLeaseInfo']['resourceInfoList']))
        if len(available_machines) == 0:
            logger.warning("No suitable machine found")
            return []

        active_leases = list(filter(
            lambda l: l['status'] in ['ACTIVE', 'PENDING'],
            machines_schedule['resourceLeaseInfo']['leaseInfoList']))

        machines = {}
        for machine in available_machines:
            logger.debug("Resource id %s - %s" %
                         (machine['resourceId'], machine['resourceName']))
            machines[machine['resourceId']] = machine

        for lease in active_leases:
            lease_resource_id = lease['resourceId']
            if lease_resource_id not in machines:
                logger.debug("Resource not found")
                continue

            logger.debug('Lease %s for resource %s is from %s to %s',
                         lease_resource_id,
                         machines[lease_resource_id]['resourceName'],
                         lease['startTime'],
                         lease['endTime'])
            resource_id = lease['resourceId']
            lease_start_time = datetime.strptime(
                lease['startTime'], '%Y-%m-%dT%H:%M:%S%z')
            lease_start_time.astimezone(timezone.utc)

            lease_end_time = datetime.strptime(
                lease['endTime'], '%Y-%m-%dT%H:%M:%S%z')
            lease_end_time.astimezone(timezone.utc)

            if (start_date < lease_start_time < end_date) or \
                    (start_date < lease_end_time < end_date):
                logger.info('%s is not suitable due to existing leases' %
                            machines[resource_id]['resourceName'])
                machines.pop(resource_id)

        return list(machines.values())

    def login(self, username, password):
        """
        Login to Colossus server
        """
        input_data = {
            'userName': username,
            'password': password,
            'domain': 'LDAP',
            'grantType': 'authorizationCode',
            'serviceName': 'paas'
        }

        logger.info("Authenticating %s...", username)
        _, result = self.post(Colossus.server, "/api/v1/token", input_data)
        Colossus.authentication_token = result

        if len(result['authToken']) > 0:
            logger.info("Login successful")
            return True
        else:
            logger.error("Login failed")
            return False

    def lease(self, **kwargs):
        """
        Places a lease request for a machine with Colossus
        """
        machine_schedule = self._fetch_machine_schedules(gpu_type=kwargs['gpu_type'],
                                                         poolname=kwargs['poolname'])
        start_date = datetime.now(timezone.utc)
        end_date = start_date + timedelta(hours=8)
        end_date.astimezone(timezone.utc)

        logger.info('Looking for machines to be leased from %s to %s' %
                    (start_date, end_date))
        idle_machines = self._idle_machine(
            machine_schedule, start_date, end_date)

        if len(idle_machines) == 0:
            logger.error("No suitable machine found")
            return None

        logger.info("%d suitable machines found" % len(idle_machines))

        desired_machine = idle_machines[0]
        logger.info("Leasing resource %s" % desired_machine['resourceName'])

        start_date = datetime.utcnow()
        end_date = start_date + timedelta(hours=8)
        input_data = {"os": kwargs['os'],
                      "leasingJustification": "DrugDiscovery Verification",
                      "resourceId": desired_machine['resourceId'],
                      "startTime": start_date.isoformat(),
                      "endTime": end_date.isoformat(),
                      "provisionMode": "CLEAN",
                      "ansibleGitScmUrl": "https://gitlab-master.nvidia.com/genomics/playbooks.git",
                      "ansibleGitBranch": "main",
                      "ansiblePlaybook": "drugdiscovery/ansible-nvidia-driver.yml"}

        headers = {'authorization':
                   'Bearer ' + Colossus.authentication_token['authToken']}

        logger.info(input_data)
        status, lease_data = self.post(Colossus.server,
                                       "/v3/lease",
                                       input_data,
                                       headers,
                                       202)
        logger.info(lease_data)
        logger.info(status)

    def post(self, server, url, data, input_headers={}, expected_status=200):
        """
        Place a POST request to an HTTP server, with all boiler-plate tasks needed
        pre and post invocation.
        """
        headers = {
            'Content-type': 'application/json',
            'Accept': '*/*',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-Mode': 'cors'
        }
        headers.update(input_headers)

        logging.info(str(headers))

        conn = http.client.HTTPSConnection(server,
                                           context=ssl._create_unverified_context())
        conn.request("POST", url, json.dumps(data), headers)

        response = conn.getresponse()
        logger.info("Request to %s returned %s", url, response.status)

        if response.status is not expected_status:
            return response.status, None

        result_data = response.read()

        result = json.loads(result_data.decode("utf-8"))
        return response.status, result



if __name__ == "__main__":

    parser = ArgumentParser(description='Colossus lease manager')
    parser.add_argument('--os',
                        dest='os',
                        type=str,
                        default='ubuntu-18.04.5-desktop-uefi',
                        help='Operating system to be installed')
    parser.add_argument('--gpu',
                        dest='gpu',
                        type=str,
                        required=True,
                        help='Type of GPU required in the machine')
    parser.add_argument('--poolname',
                        dest='poolname',
                        type=str,
                        default='app-validation',
                        help='Type of GPU required in the machine')
    args = parser.parse_args()

    # Accept user credentials
    username = getpass.getuser()
    username = input(f'Username({username}): ') or username
    password = getpass.getpass(f'Password({username}): ')

    c = Colossus()
    c.login(username, password)
    c.lease(os=args.os,
            gpu_type=args.gpu,
            poolname=args.poolname)
