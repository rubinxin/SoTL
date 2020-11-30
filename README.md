# Revisiting the Train Loss: an Efficient Performance Estimator for Neural Architecture Search

This is the code repository for our paper.

## Requirements

To install the following dependencies:
 - Python >= 3.6.0
 - scikit-learn 
 - nas-bench-201==1.3

Download the NAS-Bench-201 dataset from [here](https://github.com/D-X-Y/NAS-Bench-201])
and put it in the current folder.

## Prestore data for 5000 valid architectures
```
python prestore_arch_data.py
```

## Compare rank correlation performance of various performance estimators 
```
python rank_correlation_comparison.py
```