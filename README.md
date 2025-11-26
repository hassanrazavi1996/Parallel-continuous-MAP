Instructions for Installing

First, clone the resipotory:
%gh repo clone hassanrazavi1996/Parallel-continuous-MAP

Make conda environment and install the package:
% conda create --name pcmap python=3.9
% conda activate pcmap
% cd Parallel-continuous-MAP
% pip install .


You can also install jupyter notebook: 
conda install jupyterlab
% conda install -c anaconda ipykernel
% python -m ipykernel install --user --name=pcmap
% jupyter-lab notebooks/ &
