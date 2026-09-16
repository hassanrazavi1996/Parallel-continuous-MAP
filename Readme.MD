The codes are implementations of the parallel-in-time methods in the following paper:
Hassan Razavi, Ángel F. García-Fernández, and Simo Särkkä. "Temporal parallelisation of continuous-time maximum-a-posteriori trajectory estimation." arXiv preprint arXiv:2512.13319 (2025). [The arXive's link](https://arxiv.org/abs/2512.13319)


## Instructions for Installing
Clone the repository
```bash
git clone git@github.com:hassanrazavi1996/Parallel-continuous-MAP.git
```

Create & activate Conda environment, then install the package
```bash
conda create --name pcmap python=3.10
conda activate pcmap
cd Parallel-continuous-MAP
pip install .
```

Run in JupyterLab
```bash
conda install jupyterlab
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=pcmap
jupyter-lab notebooks/ &
```

