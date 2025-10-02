# syntax=docker/dockerfile:1.7
FROM --platform=$TARGETPLATFORM tensorflow/tensorflow:2.20.0

RUN pip install --no-cache-dir numpy scipy matplotlib tqdm

# Set working directory
WORKDIR /app

# Copy project files (if building from local repo)
# COPY . /app/

# Or clone from GitHub
# RUN git clone https://github.com/greydanus/crypto-rnn.git /app

CMD ["/bin/bash"]