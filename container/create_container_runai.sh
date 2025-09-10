#!/bin/bash

runai delete job <jobname>

runai submit \
  --name <jobname> \
  --image "registry.rcp.epfl.ch/imos-<user>/<container_name>:1.0.0" \
  --gpu 1 \
  --cpu 32 \
  --memory 100Gi \
  --run-as-uid <uid> \
  --run-as-gid <gid> \
  --supplemental-groups <gid> \
  --existing-pvc "claimname=imos-scratch,path=/mnt" \
  --large-shm \
  --command -- sleep infinity