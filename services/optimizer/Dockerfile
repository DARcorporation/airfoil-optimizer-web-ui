FROM danieldv/hode:latest
LABEL maintainer="D. de Vries <daniel.devries@darcorop.com>"
LABEL description="Parallel GA airfoil optimizer"

USER root
WORKDIR /tmp

# Install stuff using apt-get
RUN apt-get -qq update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get -y install nano && \
    apt-get -y purge cmake && \
    apt-get -y autoremove && \
    pip3 install 'cmake==3.12.*'

# Install requirements with pip
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade -r requirements.txt

# Install XFOIL
RUN wget -O xfoil.tar.gz https://github.com/daniel-de-vries/xfoil-python/archive/1.1.0.tar.gz && \
    tar -xzf xfoil.tar.gz && \
    pip3 install ./xfoil-python-1.1.0 && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Get airfoil-optimizer from Git
RUN mkdir airfoil-optimizer && \
    wget -O - https://github.com/daniel-de-vries/airfoil-optimizer/archive/0.9.0.tar.gz \
        | tar xz -C airfoil-optimizer --strip-components 1 && \
    pip3 install ./airfoil-optimizer && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#COPY Airfoil_Optimizer-0.8.0-py3-none-any.whl /usr/src/app
#RUN pip3 install Airfoil_Optimizer-0.8.0-py3-none-any.whl

# Add source and switch working directory to it
COPY . /usr/src/app
WORKDIR /usr/src/app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# run server
CMD python3 -u runner.py
