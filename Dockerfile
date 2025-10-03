# For Apple Silicon, pass --platform=linux/amd64 to docker build/run
FROM python:2.7-slim

# Use archived Debian repos (EOL) so apt works
RUN set -eux; \
  sed -i -e 's|deb.debian.org/debian|archive.debian.org/debian|g' \
         -e 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' \
         /etc/apt/sources.list; \
  printf 'Acquire::Check-Valid-Until "false";\nAcquire::AllowInsecureRepositories "true";\n' \
         > /etc/apt/apt.conf.d/99archive; \
  apt-get -o Acquire::Check-Valid-Until=false update; \
  apt-get install -y --no-install-recommends \
    build-essential \
    libzmq3-dev \
    git \
    vim \
    wget \
    ca-certificates; \
  rm -rf /var/lib/apt/lists/*

# Install the ORIGINAL versions as specified in the paper
# TensorFlow 1.1 + compatible NumPy
RUN pip install --no-cache-dir \
    tensorflow==1.1.0 \
    numpy==1.12.1 \
    matplotlib==2.0.0

# --- crypto-enigma (Py2.7) from source ---
RUN pip install 'enum34<2' 'cachetools<4'
RUN git clone --depth=1 https://github.com/orome/crypto-enigma-py.git /opt/crypto-enigma \
 && cd /opt/crypto-enigma \
 && python setup.py install \
 && cd / && rm -rf /opt/crypto-enigma    

# Set working directory
WORKDIR /app

# Copy project files (if building from local repo)
# COPY . /app/

# Or clone from GitHub
# RUN git clone https://github.com/greydanus/crypto-rnn.git /app

CMD ["/bin/bash"]
