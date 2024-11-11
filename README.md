# MB-Conjugate-Gradient-DP
Implementation of Mini-Batch Conjugate Gradient with Data Persistency (MBCG-DP) proposed in

[Lapucci M. and Pucci D. - Effectively Leveraging Momentum Terms in Stochastic Line Search Frameworks for Fast Optimization of Finite-Sum Problems - arXiv pre-print (2024)](
https://arxiv.org/)

## Installation

In order to execute the code, you will need a working [Anaconda](https://www.anaconda.com/) installation. We suggest the creation of a new conda environment with ```Python 3.11.3``` or above.
Requirements can be installed through the following:
```
pip install -r requirements.txt
```

## Usage
In order to execute the experiments, run the following:
```
python trainval.py [options]
```
The following arguments shall be specified:

| Short Option  | Long Option        | Type    | Description                                          | Default           |
|---------------|--------------------|---------|------------------------------------------------------|-------------------|
| `-e`          | `--exp_group_list` | `str`   | List of experiments to be executed                   | None (required)   |
| `-d`          | `--datadir`        | `str`   | Path to save the datasets (downloaded automatically) | None (required)   |
| `-sb`         | `--savedir_base`   | `str`   | Path to save the output logs                         | None (required)   |

The file ```exp_configs.py``` allows to specify the configurations of the experiments.

An example usage is the following:
```
python trainval -e mushrooms -d Datasets -sb Results
```

## Acknowledgements
Our implementation exploits the experimental framework used in 

In case you employed our code for research purposes, please cite the following:

```
```
