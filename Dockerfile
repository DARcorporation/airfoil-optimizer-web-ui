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
RUN pip3 install --upgrade 'numpy<1.18,>=1.17' 'scipy<1.4,>=1.3' 'openmdao<2.9,>=2.8' 'tqdm<5,>=4.32' matplotlib py3DNS validate_email

# Install XFOIL
RUN wget -O xfoil.tar.gz https://github.com/daniel-de-vries/xfoil-python/archive/1.0.3.tar.gz && \
    tar -xzf xfoil.tar.gz && \
    pip3 install ./xfoil-python-1.0.3 && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Modify /usr/local/lib/python3.6/dist-packages/openmdao/drivers/genetic_algorithm_driver.py with tqdm
COPY genetic_algorithm_driver.py /usr/local/lib/python3.6/dist-packages/openmdao/drivers/

# Add PDOT source and switch working directory to it
COPY cst.py problem.py util.py naca0012.dat runner.py /af-opt/
WORKDIR /af-opt

#
CMD python3 -u runner.py
