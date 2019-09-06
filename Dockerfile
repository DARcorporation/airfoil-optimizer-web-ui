FROM danieldv/hode:latest as pdot
LABEL maintainer="D. de Vries <daniel.devries@darcorop.com>"
LABEL description="Propeller Design Optimization Tool (PDOT)"

USER root
WORKDIR /tmp

# Uninstall system CMake, install nano, and install newer version using pip
RUN apt-get -qq update && \
    apt-get -y install nano && \
    apt-get -y purge cmake && \
    apt-get -y autoremove && \
    pip3 install 'cmake>=3.12'

# Install requirements with pip
RUN pip3 install --upgrade 'numpy>=1.17,<1.18' 'scipy>=1.3,<1.4' 'openmdao>=2.8,<2.9' 'tqdm>=4.32,<4.33'

# Install XFOIL
RUN wget -O xfoil.tar.gz https://github.com/daniel-de-vries/xfoil-python/archive/1.0.3.tar.gz && \
    tar -xzf xfoil.tar.gz && \
    pip3 install ./xfoil-python-1.0.3 && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Modify /usr/local/lib/python3.6/dist-packages/openmdao/drivers/genetic_algorithm_driver.py with tqdm
COPY genetic_algorithm_driver.py /usr/local/lib/python3.6/dist-packages/openmdao/drivers/

# Add PDOT source and switch working directory to it
COPY cst.py problem.py util.py naca0012.dat /af-opt/
WORKDIR /af-opt

#
ENV np=10
CMD mpirun -np $np python3 problem.py
