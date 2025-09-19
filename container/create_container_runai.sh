#!/bin/bash
export RCP_CAAS_GASPAR=<gaspar>
export RCP_CAAS_UID=<uid
export RCP_CAAS_GID=<gid>
export RCP_CAAS_LABSCRATCH=imos-scratch
export RCP_CAAS_EXTRAGROUP=<gid>
export RCP_CAAS_IMAGE=registry.rcp.epfl.ch/<lab>-<gaspar>/tabphm:1.0.0
export RCP_CAAS_GPU=1
export RCP_CAAS_CPU=32
export RCP_CAAS_MEMORY=100Gi

export CONTAINER_NAME=tabphm-1

runai delete job ${CONTAINER_NAME}

runai submit \
  --name ${CONTAINER_NAME} \
  --image ${RCP_CAAS_IMAGE} \
  --environment HOME="/home/${RCP_CAAS_GASPAR}" \
  --environment WANDB_API_KEY=SECRET:wandb-secret,secret1 \
  --gpu ${RCP_CAAS_GPU} \
  --cpu ${RCP_CAAS_CPU} \
  --memory ${RCP_CAAS_MEMORY} \
  --run-as-uid ${RCP_CAAS_UID} \
  --run-as-gid ${RCP_CAAS_GID} \
  --supplemental-groups ${RCP_CAAS_EXTRAGROUP} \
  --existing-pvc "claimname=${RCP_CAAS_LABSCRATCH},path=/mnt" \
  --large-shm \
  --command -- sleep infinity
