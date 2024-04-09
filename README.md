# 6Sense: Internet-Wide IPv6 Scanning and its Security Applications

## Setup & Run

Please follow this order to set up each component.

### Scanner

The scanner used by 6Sense can be found here: https://github.com/IPv6-Security/scanv6

Download and install it somewhere on your machine. To check that it compiled properly, run `./scanv6 -h`. 6Sense will create the config file for scanv6 based on the parameters in the 6Sense config.py. You can optionally define your config file separately and pass it to 6Sense by setting the config file path and setting RESET_CONFIG to be False. 

### Offline Dealiaser

The offline dealiaser (used for prefix matching of aliased regions) used by 6Sense is here: https://github.com/IPv6-Security/offline-dealiaser

Download and install it somewhere on your machine. To check that it compiled properly, run `./aliasv6 -h`.

### Datasets
You will need three separate datasets to run 6Sense:

1. A mapping of IPv6 routing prefixes to ASes. We suggest Routeviews Pfx2AS for ease of use. It can be found here: https://www.caida.org/catalog/datasets/routeviews-prefix2as/
2. A list of known aliased prefixes. One option is the alias prefix dataset provided by the IPv6 Hitlist here: https://ipv6hitlist.github.io/. You may use an empty file if you wish to bypass offline dealiasing. 
3. A seed dataset of potentially active IPv6 addresses. Choosing the optimal seed dataset for a particular use case is a nontrivial open problem. Hitlists such as the IPv6 Hitlist (https://ipv6hitlist.github.io/) give some diversity of data sources, and may be good for early users. Choose your seed dataset wisely as it will determine what you find...


### Generator

When both the Scanner and the Dealiaser are built, and you have your dataset, you are ready to run 6Sense! Follow the steps below to setup the generator.

1. Set all of the file paths in `config.py` (including the paths to the scanner and dealiaser you just installed and the datasets you just downloaded or collected). Keep in mind some paths are relative to your home directory, and some are absolute, so make sure to check the comments in `config.py` if it's not finding your filepath.
2. Set the scanner parameters in config.py. 
3. Create your python environment (we suggest using Conda) and initialize it with the packages in requirements.txt.conda (or requirements.txt.pip if using a pip-based environment).
4. Build the cython module for printing IPs with  `python3 setup.py build_ext --inplace` Be careful to do this in your final python envionrment. Cython does not port well between different python enivronments. 
5. Next you need to ensure your pfx2as file is properly parsed. You can do so by running `python3 rounding.py pfx2as_filename.dat pfx2as_rounded_filename.dat`.
6. Initiate a Jupyter Notebook instance for `Gradient_Testing_On_Generation.ipynb`.
7. To train the models, you should run the cells under Train Model.
8. To run the model, you should run the cells under Run Model.


## License

6Sense is licensed under Apache 2.0. For more information, see the LICENSE file.

