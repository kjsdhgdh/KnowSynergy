# Requirements
* python 3.7
* deepchem >= 2.5
* numpy >= 1.19
* pandas >= 1.3
* pytorch >= 1.8.0
* pytorch geometric >= 2.0.0 
* scikit-learn >= 1.0.2
* rdkit >= 2020.09.1

# Usage
```sh
  cd ./src
  # for regression experiment
  python main_cl_joint_reg.py --dataset=ALMANAC --cv_mode=1 
  #--dataset=ONEIL, --cv_mode=2,3

  # for classification experiment
  python main_cl_joint.py --dataset=ALMANAC --cv_mode=1 
  #--dataset=ONEIL, --cv_mode=2,3
```
