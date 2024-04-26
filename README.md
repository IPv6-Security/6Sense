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

### Generator Parameters
6Sense contains numerous parameters that can be changed to augment generator behavior. While there are many variations of inputs, we'll address a few of the important parameters (specifically those passed to the `ComparisonModel_online` function and it's `Upper64_HPs` and `Lower64_HPs` parameters) here:

- **Generation Amount**: The first and primary input parameter to `ComparisonModel_online` is the number of addresses to generate.
- **Comparison Name**: The second input parameter to `ComparisonModel_online` is the experiment name (used when creating output files/folders).
- `allocation_gradient_threshold`: This parameter adjusts how long the Breadth phase (even sampling) of allocation generation lasts. After generating every `allocation_gradient_amount` addresses, 6Sense checks the ratio of new allocations discovered with active addresses in that batch to total allocations discovered with active addreses so far. Once this ratio drops below `allocation_gradient_threshold`, allocation generation switches to Depth. If not provided, it defaults to `0.005`. This default provides a good depth/breadth ratio for a `100M` address scan. It is suggested to decrease this parameter as you increase the number of addresses to generate (since you have more budget to explore initially). One suggestion is to decrease this parameter by an order of magnitude with every order of magnitude increase in generative amount. 
- `allocation_gradient_amount`: Used in conjunction with `allocation_gradient_threshold` to describe how many IPs to generate before re-evaluating the number of new ASes found. If not provided it defaults to `1M`.
- `ppi`: Number of addresses to generate per Depth iteration before updating the Allocation weights. By default it is `1M`. We suggest increasing this when generation amount increases by an order of magnitude (i.e. increasing from `1M` for a `100M` scan to `10M` for a `1B` scan) to ensure Depth allocation sampling does not converge too quickly.
- `per_iteration`: How many IPs to generate before sending to the scanner. This is bounded by `allocation_gradient_threshold` for the Breadth phase and `ppi` for the Depth phase. This does not change generator behavior, but can optimze for efficiency/concurrency (i.e. if scanning slowly, it may make sense to generate smaller amounts like `100K` at a time, so more scanning can occur concurrently with generation). Important for optimization because generating larger batches is typically faster (i.e. due to GPU optimizations with the LSTM and moving data between modules), but smaller batches can be sent to the scanner more quickly. We suggest using `100K` for most applications, since this ensures most scanning is concurrent with generation (since usually generation is faster than scanning), but if your scan rate is very high, you may wish to change this to `500K` or higher.
- `Upper64_HPs`: Hyperparameters describing the LSTM. For most users we do not suggest adjusting these. However, users on multi-gpu machines can adjust the number of gpus used during generation with the `gpus` parameter.
- `Lower64_HPs.subprocesses`: The `subprocesses` attribute allows users to list how many subprocesses to allocate to the Multiprocess Lower-64 generator.

## License

6Sense is licensed under Apache 2.0. For more information, see the LICENSE file.

