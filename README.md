# Semantic segmentation | SPBSTU [lab_work]

## Project organization

    ├── LICENSE
    ├── README.md          <- README file
    ├── data
    │   ├── external       <- Data from third party sources. [Use for predict, wich will add in future version]
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    |
    ├── imports.py         <- all libraries for project
    ├── make_dataset.py    <- response by work with dataset
    ├── net.py             <- contain class of model
    ├── pipeline.py        <- pipeline from load dataset to get results
    ├── train.py           <- train function
    ├── visualize.py       <- visualize the result

## Getting started

+ Download `train`, `train_mask` from kaggle: [Carvana Image Msking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data) and put to data/raw
+ Load requirements: `pip install -r requirements.txt`
+ Run: `python3 pipeline.py`


## License
MIT

## Project status
Development is continuing
