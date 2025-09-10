#####################################
# RCP CaaS requirement (Image)
#####################################
# FROM pytorch/pytorch@sha256:f894dae26e1ee8557c544f9cfdb9dc011b1552bf3c1e656b422f2e221d380e40
FROM python:3.12-slim-trixie


#####################################
# System packages (must be as root)
#####################################
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Zurich

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    git wget curl bash zip unzip iputils-ping vim htop libgl1 tmux screen \
    g++ gcc ffmpeg libsm6 libxext6 \
    libosmesa6-dev freeglut3-dev libglfw3 libgles2-mesa-dev \
    syncthing rsync rclone tzdata sudo; \
    rm -rf /var/lib/apt/lists/*



#####################################
# Install uv system-wide (root during build only)
#####################################
# Pin a uv version for reproducibility. Avoid :latest.
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /usr/local/bin/


COPY pyproject.toml .
RUN uv pip install -r pyproject.toml --system
# COPY . .
# RUN uv pip install -e .


#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}


WORKDIR /home/${LDAP_USERNAME}

# Copy your code inside the container
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Create workspace directory
RUN mkdir -p /workspace \
    && chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /workspace \
    && chmod -R 775 /workspace

USER ${LDAP_USERNAME}

RUN ln -s /mnt/imos-students/${LDAP_USERNAME}/experiments /workspace && \
    ln -s /mnt/imos-students/${LDAP_USERNAME}/datasets /workspace && \
    ln -s /mnt/imos-students/${LDAP_USERNAME}/projects /workspace

WORKDIR /workspace

