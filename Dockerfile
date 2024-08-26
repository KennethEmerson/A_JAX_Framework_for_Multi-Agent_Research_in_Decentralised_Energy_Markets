########################################################################################################################
# A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
########################################################################################################################
# author: K. Emerson
# Department of Computer Science
# Faculty of Sciences and Bioengineering Sciences
# Vrije Universiteit Brussel
########################################################################################################################
# Dockerfile configuration for running the framework in a Microsoft Visual Studio Code development container
########################################################################################################################

ARG APP_NAME=main
ARG APP_PATH=code
ARG PYTHON_VERSION=3.11.0
ARG POETRY_VERSION=1.7.1


########################################################################################################################
# BASE IMAGE
########################################################################################################################

FROM python:$PYTHON_VERSION-slim-bullseye as base
# might be required to install specific dependencies which require compiling
RUN apt-get update -y
RUN apt-get install gcc g++ python3-dev -y


########################################################################################################################
# DEVELOPMENT
########################################################################################################################
FROM base as development
ARG APP_NAME
ARG APP_PATH
ARG POETRY_VERSION

# install ssh server if required tot connect to ssh based repos
RUN apt-get install openssh-server -y

# install git inside the container
RUN export DEBIAN_FRONTEND=noninteractive && apt-get -y install --no-install-recommends git 

# install poetry
RUN pip install --upgrade pip poetry==$POETRY_VERSION 

# install dependencies
COPY ./$APP_PATH/pyproject.toml ./$APP_PATH/poetry.lock ./
ENV POETRY_VIRTUALENVS_CREATE=false
RUN poetry install 


