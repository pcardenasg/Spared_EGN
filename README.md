# Spared_EGN
EGN state-of-the-art model adapted for [SPARED](https://arxiv.org/abs/2407.13027#) datasets.

## Environment set up
Create environment:
```
cd Spared_EGN
conda env create -f environment.yml -n egn_spared
conda activate egn_spared
pip install -r requirements.txt
```

## Running the complete EGN framework (exemplar building + gene expression prediction)
```
python run_full_egn.py --dataset [SPARED_dataset_name] --lr [learning_rate]
```

## Running each EGN phase separately
* Building the exemplars:
```
python exemplars.py --dataset [SPARED_dataset_name]
```
* Training and testing the gene expression prediction model:
```
python main.py --dataset [SPARED_dataset_name] --lr [learning_rate]
```