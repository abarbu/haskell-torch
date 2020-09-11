FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg ca-certificates dirmngr && \
    rm -rf /var/lib/apt/lists/*

ARG GHC=8.10.1
ARG DEBIAN_KEY=427CB69AAC9D00F2A43CAF1CBA3CBA3FFE22B574
ARG CABAL_INSTALL=3.2
ARG STACK=2.3.1
ARG STACK_KEY=C5705533DA4F78D8664B5DC0575159689BEFB442
ARG STACK_RELEASE_KEY=2C6A674E85EE3FB896AFC9B965101FF31C5C154D

ENV PATH /opt/conda/bin:/root/.cabal/bin:/root/.local/bin:/opt/conda/bin/:/opt/cabal/${CABAL_INSTALL}/bin:$PATH

RUN export GNUPGHOME="$(mktemp -d)" && \
    echo "disable-ipv6" >> ${GNUPGHOME}/dirmngr.conf && \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys ${DEBIAN_KEY} && \
    gpg --batch --armor --export ${DEBIAN_KEY} > /etc/apt/trusted.gpg.d/haskell.org.gpg.asc && \
    gpgconf --kill all && \
    echo 'deb http://downloads.haskell.org/debian buster main' > /etc/apt/sources.list.d/ghc.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
	cabal-install-${CABAL_INSTALL} \
        wget \
        curl \
        g++ \
        git \
        make \
        netbase \
        openssh-client \
        xz-utils \
	zlib1g-dev \
        libgmp3-dev \
	git \
        patch \
	bzip2 \
	protobuf-compiler \
	libprotoc-dev \
	emacs25-lucid \
	vim \
	less \
	procps \
	gdb \
	screen && \
    rm -rf "$GNUPGHOME" /var/lib/apt/lists/*

RUN export GNUPGHOME="$(mktemp -d)" && \
    # echo "disable-ipv6" >> ${GNUPGHOME}/dirmngr.conf && \
    # gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys ${STACK_KEY} && \
    # gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys ${STACK_RELEASE_KEY} && \
    curl -fSL https://github.com/commercialhaskell/stack/releases/download/v${STACK}/stack-${STACK}-linux-x86_64.tar.gz -o stack.tar.gz && \
    # curl -fSL https://github.com/commercialhaskell/stack/releases/download/v${STACK}/stack-${STACK}-linux-x86_64.tar.gz.asc -o stack.tar.gz.asc && \
    # gpg --batch --trusted-key 0x575159689BEFB442 --verify stack.tar.gz.asc stack.tar.gz && \
    tar -xf stack.tar.gz -C /usr/local/bin --strip-components=1 && \
    rm -rf "$GNUPGHOME" /var/lib/apt/lists/* /stack.tar.gz.asc /stack.tar.gz && \
    stack setup

RUN cd /tmp/ && \
    git clone --recurse-submodules https://github.com/abarbu/haskell-torch.git && \
    cd haskell-torch

# TODO Remove me
COPY setup.sh /tmp/haskell-torch/setup.sh
COPY generate-config.py /tmp/haskell-torch/generate-config.py

RUN eval "$(/opt/conda/bin/conda shell.bash hook)" && \
    conda activate base && \
    cd /tmp/haskell-torch && \
    bash ./setup.sh --with-jupyter --with-cuda --in-conda-base && \
    cat stack.yaml

RUN eval "$(/opt/conda/bin/conda shell.bash hook)" && \
    cd /tmp/haskell-torch && \
    stack ghci --ghc-options -fobject-code <<< ""

RUN echo '#!/bin/bash\ncd /tmp/haskell-torch && eval "$(/opt/conda/bin/conda shell.bash hook)" && stack ghci --ghc-options -fobject-code' > /tmp/start-ghci && \
    chmod +x /tmp/start-ghci

RUN echo '#!/bin/bash\ncd /tmp/haskell-torch && eval "$(/opt/conda/bin/conda shell.bash hook)" && stack ghci --no-build --ghc-options -fobject-code' > /tmp/start-ghci-no-build && \
    chmod +x /tmp/start-ghci-no-build

RUN echo '#!/bin/bash\n cd /tmp/haskell-torch && eval "$(/opt/conda/bin/conda shell.bash hook)" && stack exec jupyter -- notebook --ip=0.0.0.0 --port=8080 --allow-root --no-browser' > /tmp/start-notebook && \
    chmod +x /tmp/start-notebook

RUN echo '#!/bin/bash\n cd /tmp/haskell-torch && eval "$(/opt/conda/bin/conda shell.bash hook)" && stack exec jupyter -- console --kernel=Haskell' > /tmp/start-console && \
    chmod +x /tmp/start-console

CMD ["/tmp/start-notebook"]
