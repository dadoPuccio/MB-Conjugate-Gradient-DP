# MB-Conjugate-Gradient-DP
Implementation of Mini-Batch Conjugate Gradient with Data Persistency (MBCG-DP) proposed in

[Lapucci M. and Pucci D. - Effectively Leveraging Momentum Terms in Stochastic Line Search Frameworks for Fast Optimization of Finite-Sum Problems - arXiv pre-print (2024)](https://arxiv.org/abs/2411.07102)

## Installation

In order to execute the code, you will need a working [Anaconda](https://www.anaconda.com/) installation. We suggest the creation of a new conda environment with ```Python 3.11.3``` or above.
Requirements can be installed through:
```
pip install -r requirements.txt
```

## Usage
In order to run the experiments, execute the following:
```
python trainval.py [options]
```
The following arguments shall be specified:

<div align='center'>
  
| Short Option  | Long Option           | Type    | Description                                          | Default           |
|---------------|-----------------------|---------|------------------------------------------------------|-------------------|
| `-e`          | `--exp_group_list`    | `str`   | List of experiments to be executed                   | None (required)   |
| `-d`          | `--datadir`           | `str`   | Path to save the datasets (downloaded automatically) | None (required)   |
| `-sb`         | `--savedir_base`      | `str`   | Path to save the output logs                         | None (required)   |

</div>

The file ```exp_configs.py``` specifies the experiments avaialable and their configurations.

An example usage is:
```
python trainval -e mushrooms -d Datasets -sb Results
```

The ```PlotResults``` functionalities enable the generation of plots based on logs produced during the experiments.

## Acknowledgements
Our experimental framework is a simplified version of that of [SLS](https://github.com/IssamLaradji/sls) and [PoNoS](https://github.com/leonardogalli91/PoNoS).

In case you employed our code for research purposes, please cite:

```
@misc{lapucci2024effectivelyleveragingmomentumterms,
      title={Effectively Leveraging Momentum Terms in Stochastic Line Search Frameworks for Fast Optimization of Finite-Sum Problems}, 
      author={Matteo Lapucci and Davide Pucci},
      year={2024},
      eprint={2411.07102},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2411.07102}, 
}
```
