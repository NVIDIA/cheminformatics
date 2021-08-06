# Setting up development environment

- Start container in dev mode
   ```
   cd .. && ./launch.sh dev
   ```

- To start database server, run the following commands:
   ```
   cd chemportal/db
   ./launch reset
   ```

- Inside the dev container, run the following commands:
   ```
   cd /workspace/chemportal
   python3 -m cuchemportal
   ```

- Start frontend server
   ```
   cd chemportal/frontend
   npm run serve
   ```

# Run complete suite
WIP - Currently the starup script does not support it.
