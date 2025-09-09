#####################################
# RCP CaaS requirement (Image)
#####################################
FROM pytorch/pytorch@sha256:f894dae26e1ee8557c544f9cfdb9dc011b1552bf3c1e656b422f2e221d380e40


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
COPY environment.yml /home/${LDAP_USERNAME}
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Create workspace directory
RUN mkdir -p /workspace \
    && chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /workspace \
    && chmod -R 775 /workspace

USER ${LDAP_USERNAME}

RUN ln -s /mnt/imos-students/${LDAP_USERNAME}/experiments /workspace && \
    ln -s /mnt/imos-students/${LDAP_USERNAME}/datasets /workspace && \
    ln -s /mnt/imos-students/${LDAP_USERNAME}/projects /workspace

RUN conda env create -f environment.yml

ENV PATH=/opt/conda/envs/IMOS_Lab/bin:$PATH
SHELL ["/bin/bash","-l","-c"]

WORKDIR /workspace

RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate IMOS_Lab" >> ~/.bashrc

