# ref https://github.com/tebeka/pythonwise/blob/master/docker-miniconda/Dockerfile
FROM ubuntu:18.04

# System packages 
RUN apt-get update && apt-get install -y curl
RUN apt-get install -y git

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Python packages from conda
RUN conda install -c anaconda -y python=3.7.2
RUN conda install -c anaconda -y pip

RUN conda install -c pytorch -y \
    pytorch \
    torchvision \
    cpuonly
RUN conda install -c conda-forge -y opencv
RUN pip install cython
RUN apt-get -y install gcc
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN git clone https://github.com/ethanweber/xview.git
WORKDIR /xview
RUN apt-get install -y wget
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uw3BiKYq5dzYVOSj_pyPUAPrB8F4Lwh8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uw3BiKYq5dzYVOSj_pyPUAPrB8F4Lwh8" -O model_weights.pth && rm -rf /tmp/cookies.txt
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nM76Kn7w7I-NL4xljbkNpWU4_TgqmAHj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uw3BiKYq5dzYVOSj_pyPUAPrB8F4Lwh8" -O hassan_model_weights.pth && rm -rf /tmp/cookies.txt
RUN git submodule init && git submodule update
# ARG INCUBATOR_VER=unknown
WORKDIR /xview/detectron2_repo
RUN conda install -c anaconda scikit-image
RUN apt-get install -y g++
RUN python setup.py build develop
WORKDIR /xview
ADD test_pre_00123.png /test_pre_00123.png
ADD test_post_00123.png /test_post_00123.png
RUN conda install -y pillow=6.1
RUN pip install -U git+https://github.com/facebookresearch/fvcore.git
RUN conda install -c anaconda -y scikit-image
RUN apt-get install -y vim
WORKDIR /
RUN mkdir temp
WORKDIR /xview
ENTRYPOINT ["python", "wrapper_hassan.py"]
