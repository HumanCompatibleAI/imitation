#!/bin/bash

# if port is changed here, it should also be changed in runners/launch_docker-dev.sh
cd / && jupyter lab --ip 0.0.0.0 --allow-root --no-browser --port=9988