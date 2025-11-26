## Instructions for Installing
Clone the repository
```bash
git clone git@github.com:hassanrazavi1996/Parallel-continuous-MAP.git
```

Create & activate Conda environment, then install the package
```bash
conda create --name pcmap python=3.9
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

