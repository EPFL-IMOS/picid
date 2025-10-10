#!/bin/bash

docker login registry.rcp.epfl.ch
docker push registry.rcp.epfl.ch/imos-<user>/<container_name>:1.0.1