# RandNE-Python
This is a reference Python implementation of the RandNE paper [Billion-scale Network Embedding with Iterative Random Projection](https://zw-zhang.github.io/files/2018_ICDM_RandNE.pdf) published in ICDM 2018.

Note that this is **NOT** the official implementation; to completely reproduce the results in the paper, please refer to the [official implementation](https://github.com/ZW-ZHANG/RandNE) written in MATLAB.

This code run under Python 3.

# Installation
Install RandNE and the other requirements:

	git clone https://github.com/GTmac/RandNE-Python.git
	cd RandNE-Python
	pip install -r requirements.txt

# Usage
To run RandNE on the *BlogCatalog* dataset, use the following command:

``python3 src/randne.py --input data/blogcatalog.mat --output data/blogcatalog-emb.mat --use-trans-matrix -q 3 -d 128 --weights 1 100 1000``

To check the full list of command line options, run:

``python3 src/randne.py --help``

# Evaluation
To evaluate the embeddings on the multi-label node classification task, run the following command:

``python3 src/scoring.py --emb data/blogcatalog-emb.mat --network data/blogcatalog.mat --num-shuffles 5``

Again, to check the full list of command line options, run:

``python3 src/scoring.py --help``
