FROM continuumio/anaconda3:4.1.1

# set backend for matplotlib to Agg
RUN matplotlibrc_path=$(python -c "import site, os, fileinput; packages_dir = site.getsitepackages()[0]; print(os.path.join(packages_dir, 'matplotlib', 'mpl-data', 'matplotlibrc'))") && \
    sed -i 's/^backend      : Qt4Agg/backend      : Agg/' $matplotlibrc_path

# Stop jupyter nbconvert trying to rewrite its folder hierarchy
RUN mkdir -p /root/.jupyter && touch /root/.jupyter/jupyter_nbconvert_config.py && touch /root/.jupyter/migrated && \
    mkdir -p /.jupyter && touch /.jupyter/jupyter_nbconvert_config.py && touch /.jupyter/migrated && \
    # Stop Matplotlib printing junk to the console on first load
    sed -i "s/^.*Matplotlib is building the font cache using fc-list.*$/# Warning removed by Kaggle/g" /opt/conda/lib/python3.5/site-packages/matplotlib/font_manager.py && \
    # Make matplotlib output in Jupyter notebooks display correctly
    mkdir -p /etc/ipython/ && echo "c = get_config(); c.IPKernelApp.matplotlib = 'inline'" > /etc/ipython/ipython_config.py

COPY . /pckg/asc
WORKDIR /pckg/asc
RUN pip install -e .
WORKDIR /
RUN echo "import nltk" > setup.py
RUN echo "nltk.download('all-corpora')"  >> setup.py
RUN python setup.py
RUN rm setup.py
RUN conda install -y gensim

WORKDIR /lib	
COPY GoogleNews-vectors-negative300.bin.gz GoogleNews-vectors-negative300.bin.gz
RUN gunzip GoogleNews-vectors-negative300.bin.gz
COPY freebase-vectors-skipgram1000-en.bin.gz freebase-vectors-skipgram1000-en.bin.gz
RUN gunzip freebase-vectors-skipgram1000-en.bin.gz
WORKDIR /
RUN echo "import nltk" > setup.py
RUN echo "nltk.download('punkt')"  >> setup.py
RUN echo "nltk.download('averaged_perceptron_tagger')"  >> setup.py
RUN python setup.py
RUN rm setup.py
