FROM debian:buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg ca-certificates dirmngr && \
    rm -rf /var/lib/apt/lists/*

ARG GHC=8.10.1
ARG DEBIAN_KEY=427CB69AAC9D00F2A43CAF1CBA3CBA3FFE22B574
ARG CABAL_INSTALL=3.2
ARG STACK=2.3.1
ARG STACK_KEY=C5705533DA4F78D8664B5DC0575159689BEFB442
ARG STACK_RELEASE_KEY=2C6A674E85EE3FB896AFC9B965101FF31C5C154D

ENV PATH /opt/conda/bin:/root/.cabal/bin:/root/.local/bin:/opt/anaconda/bin/:/opt/cabal/${CABAL_INSTALL}/bin:$PATH

RUN export GNUPGHOME="$(mktemp -d)" && \
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
	bzip2 \
	protobuf-compiler \
	libprotoc-dev \
	emacs25-lucid \
	vim \
	less \
	procps \
	gdb \
	screen && \
    rm -rf "$GNUPGHOME" /var/lib/apt/lists/* && \
    wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /opt/anaconda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN export GNUPGHOME="$(mktemp -d)" && \
    gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys ${STACK_KEY} && \
    gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys ${STACK_RELEASE_KEY} && \
    curl -fSL https://github.com/commercialhaskell/stack/releases/download/v${STACK}/stack-${STACK}-linux-x86_64.tar.gz -o stack.tar.gz && \
    curl -fSL https://github.com/commercialhaskell/stack/releases/download/v${STACK}/stack-${STACK}-linux-x86_64.tar.gz.asc -o stack.tar.gz.asc && \
    gpg --batch --trusted-key 0x575159689BEFB442 --verify stack.tar.gz.asc stack.tar.gz && \
    tar -xf stack.tar.gz -C /usr/local/bin --strip-components=1 && \
    rm -rf "$GNUPGHOME" /var/lib/apt/lists/* /stack.tar.gz.asc /stack.tar.gz && \
    stack setup

RUN cd /tmp/ && \
    git clone --recurse-submodules https://github.com/abarbu/haskell-torch.git && \
    cd haskell-torch && \
    git checkout 28e2ddabb38e718854a19fd677ff15c6697a3d09

RUN eval "$(/opt/anaconda/bin/conda shell.bash hook)" && \
    conda activate base && \
    cd /tmp/haskell-torch && \
    bash ./setup.sh --without-jupyter --without-cuda && \
    echo "conda activate haskell-torch" >> ~/.bashrc

RUN eval "$(/opt/anaconda/bin/conda shell.bash hook)" && \
    conda activate haskell-torch && \
    cd /tmp/haskell-torch && \
    stack ghci --ghc-options -fobject-code <<< ""

RUN echo '#!/bin/bash\ncd /tmp/haskell-torch && eval "$(/opt/anaconda/bin/conda shell.bash hook)" && conda activate haskell-torch && stack ghci --ghc-options -fobject-code' > /tmp/start-ghci && \
    chmod +x /tmp/start-ghci

RUN echo '#!/bin/bash\ncd /tmp/haskell-torch && eval "$(/opt/anaconda/bin/conda shell.bash hook)" && conda activate haskell-torch && stack ghci --no-build --ghc-options -fobject-code' > /tmp/start-ghci && \
    chmod +x /tmp/start-ghci-no-build

CMD ["/tmp/start-ghci-no-build"]
