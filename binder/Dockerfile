FROM jupyter/scipy-notebook:latest

COPY . ${HOME}

USER root

RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
