#!/bin/bash

# Fail on any error
set -e

# Required build args
LDAP_USERNAME=${LDAP_USERNAME:-<ldap_username}
LDAP_UID=${LDAP_UID:-<uid>}
LDAP_GROUPNAME=${LDAP_GROUPNAME:-imos}
LDAP_GID=${LDAP_GID:-<gid>}

# Image name
IMAGE_NAME="registry.rcp.epfl.ch/imos-<user>/<container_name>"

# Optional: Tag
TAG=1.0.1

echo "Using build arguments:"
echo "  LDAP_USERNAME   = $LDAP_USERNAME"
echo "  LDAP_UID        = $LDAP_UID"
echo "  LDAP_GROUPNAME  = $LDAP_GROUPNAME"
echo "  LDAP_GID        = $LDAP_GID"
echo "  IMAGE_NAME      = $IMAGE_NAME"
echo "  TAG             = $TAG"

# Build the image
docker build \
  --build-arg LDAP_USERNAME="$LDAP_USERNAME" \
  --build-arg LDAP_UID="$LDAP_UID" \
  --build-arg LDAP_GROUPNAME="$LDAP_GROUPNAME" \
  --build-arg LDAP_GID="$LDAP_GID" \
  --platform linux/amd64 \
  -t "$IMAGE_NAME:$TAG" \
  .

echo "Docker image '$IMAGE_NAME:$TAG' built successfully."
