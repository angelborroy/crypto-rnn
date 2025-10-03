FROM --platform=$TARGETPLATFORM tensorflow/tensorflow:2.20.0

RUN apt-get update && \ 
  apt-get install -y --no-install-recommends \
    git \
    vim; \
  rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy scipy matplotlib tqdm

# --- crypto-enigma (Py2 to Py3 quick port) ---
RUN pip install --no-cache-dir "cachetools>=4,<6" \
 && git clone --depth=1 https://github.com/orome/crypto-enigma-py.git /opt/crypto-enigma \
 && cd /opt/crypto-enigma/crypto_enigma \
 && find . -type f -name "*.py" -print0 | xargs -0 sed -i 's/\bbasestring\b/str/g' \
 && find . -type f -name "*.py" -print0 | xargs -0 sed -i 's/\bunicode\b/str/g' \
 && find . -type f -name "*.py" -print0 | xargs -0 sed -i 's/\bxrange\b/range/g' \
 && find . -type f -name "*.py" -print0 | xargs -0 sed -i 's/\.iteritems()/\.items()/g' \
 && find . -type f -name "*.py" -print0 | xargs -0 sed -i 's/\.itervalues()/\.values()/g' \
 && find . -type f -name "*.py" -print0 | xargs -0 sed -i 's/\.iterkeys()/\.keys()/g' \
 && sed -i 's/func_code/__code__/g' /opt/crypto-enigma/crypto_enigma/utils.py

# Patched machine.py for Py3 support
COPY enigma-patch/machine.py /opt/crypto-enigma/crypto_enigma/machine.py
 
# Install by copying into site-packages
RUN pysp=$(python -c "import site; print(site.getsitepackages()[0])") \
 && mkdir -p "$pysp" \
 && cp -r /opt/crypto-enigma/crypto_enigma "$pysp/crypto_enigma" \
 && rm -rf /opt/crypto-enigma

# Set working directory
WORKDIR /app

# Copy project files (if building from local repo)
# COPY . /app/

# Or clone from GitHub
# RUN git clone https://github.com/greydanus/crypto-rnn.git /app

CMD ["/bin/bash"]