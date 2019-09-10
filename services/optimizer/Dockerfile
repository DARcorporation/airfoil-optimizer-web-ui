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
    pip3 install 'cmake>=3.12'

# Install requirements with pip
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade -r requirements.txt

# Install XFOIL
RUN wget -O xfoil.tar.gz https://github.com/daniel-de-vries/xfoil-python/archive/1.0.3.tar.gz && \
    tar -xzf xfoil.tar.gz && \
    pip3 install ./xfoil-python-1.0.3 && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Modify /usr/local/lib/python3.6/dist-packages/openmdao/drivers/genetic_algorithm_driver.py with tqdm
COPY ./genetic_algorithm_driver.py /usr/local/lib/python3.6/dist-packages/openmdao/drivers/

# Add source and switch working directory to it
COPY . /usr/src/af-opt
WORKDIR /usr/src/af-opt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# run server
CMD python3 -u runner.py
