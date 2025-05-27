## Tree Detection ##

This repository provides code for training and evaluating a convolutional neural network (CNN) to detect trees / dead trees with aerial imagery. The CNN takes multispectral imagery as input and outputs a confidence map and a scale map that predict the locations and crown sizes of trees. The individual tree locations are found by local peak finding, and the tree crown sizes are determined by marker-controlled watershed segmentation of the confidence map. The algorithm takes adventage of attention mechanism and weighted focal loss to improve detection accuracy. Users can also choose amongst three popular deep learning backbones (VGG, Resnet and Efficientnet). The repository was modified from the large-scale urban tree detection algorithm (Ventura et al. 2024), with the implementation of weighted scale-adaptive heatmap regression (Luo et al. 2021). 

### Installation ###

The model is implemented with Python 3.11.4 and TensorFlow 2.18.0. You may setup the environment using:

    python -m venv env
    source env/bin/activate
    pip install tensorflow[and-cuda]==2.18.0
    pip install numpy
    pip install imageio
    pip install rasterio
    pip install geopandas
    pip install h5py
    pip install scipy
    pip install tqdm
    pip install scikit-image
    pip install scikit-learn
    pip install optuna
    pip install matplotlib

To activate the environment in the future:

    cd ~
    source env/bin/activate


Alternatively, we have provided an `environment.yml` file for earlier versions of Python and TensorFlow so that you can easily create a conda environment with the dependencies installed:

    conda env create 
    conda activate us-tree-detection


### Dataset ###

The model expects a standardized dataset and folder structure: 

    data/
    ├── images/
    │   ├── image_FL_2021_001.tif
    │   ├── image_TX_2020_023.tif
    ├── labels.gpkg

- an image folder containing images stored as .tifs
- a .gpkg containing point labels of individual trees
- the images and labels should overlap with each other geographically


First, setup your configurations (e.g. input and output directory) in the `configuration.py` script. 

To organize data for the subsequent training procedures, run the `organize.py` script. The script will automatically crop images to 256x256 patches that are required for model training.

    python3 -m organize.prepare 

Then, use the `split.py` script to create the desired train, validation, and test split of the training data. The default setting expects an independent test.txt file. To generate test data from the full dataset, use the `--generate_test`  option.

    python3 -m split.split 

Finally, to prepare a dataset for model, run the `prepare.py` script.  You can specify the bands in the input raster using the `--bands` flag (currently `RGB` and `RGBN` are supported.)

    python3 -m scripts.prepare 


### Training ###

To train the model, run the `train.py` script.

    python3 -m scripts.train

### Hyperparameter tuning ###

The model outputs a confidence map, and we use local peak finding to isolate individual trees.  We use the Optuna package to determine the optimal parameters of the peaking finding algorithm.  We search for the best of hyperparameters to maximize F-score on the validation set.

    python3 -m scripts.tune <path to hdf5 file> <path to log directory>

For example,

    python3 -m scripts.tune prepared_data.hdf5 logs

### Evaluation on test set ###

Once hyperparameter tuning finishes, use the `test.py` script to compute evaluation metrics on the test set.

    python3 -m scripts.test <path to hdf5 file> <path to log directory> --center_crop --rearrange_channels


### Inference on a large raster ###

To detect trees in rasters and produce GeoJSONs containing the geo-referenced trees, use the `inference.py` script.  The script can process a single raster or a directory of rasters.

    python3 -m scripts.inference <input tiff or directory> <output json or directory> <path to log directory> --bands <RGB or RGBN>

For example,

    python3 -m scripts.inference ../urban-tree-detection-data/image_for_inference.tif inferred_geospatial_layer.json logs

### Pre-trained weights ###

The following pre-trained models are available:

| Imagery   | Years     | Bands    | Region              | Log Directory Archive     |
|-----------|-----------|----------|---------------------|---------------------------|
| 60cm NAIP | 2018-2022 | RGBN     | US Atlantic Coast   | [OneDrive](https://cpslo-my.sharepoint.com/:u:/g/personal/jventu09_calpoly_edu/ES31TXWdeGRFj_hn3O4qZpoBfhye_ssuULyaC2WB7yaJTw?e=cYkjMf)

We also provide an [example NAIP 2020 tile from Los Angeles](https://cpslo-my.sharepoint.com/:i:/g/personal/jventu09_calpoly_edu/EU1xfporUiBDvT2ZOpW0raEBOqJcJQpqcOv1lKNMCgbCdQ?e=zsgxXs) and an [example GeoJSON predictions file from the RGBN 2016-2020 model](https://cpslo-my.sharepoint.com/:u:/g/personal/jventu09_calpoly_edu/EUHYGnWdqL5FvYc1wm9hSl8BBdL2JEgMSlqS1FiTdB0EWA?e=uZMIBc).  

You can explore a [map of predictions for the entire urban reserve of California](https://jventu09.users.earthengine.app/view/urban-tree-detector) (based on NAIP 2020 imagery) created using this pre-trained model.

### Using your own data ###

To train on your own data, you will need to organize the data into the format expected by `organize.py`.

* The code is currently designed for three-band (RGB) or four-band (red, green, blue, near-IR) imagery.  To handle more bands, you would need to add an appropriate preprocessing function in `utils/preprocess.py`.  If RGB are not in the bands, then `models/VGG.py` would need to be modified, as the code expects the first three bands to be RGB to match the pre-trained weights.
* For each image, store a csv file containing x,y coordinates for the tree locations in a file `<name>.csv` where `<name>.tif`, `<name>.tiff`, or `<name>.png` is the corresponding image. The csv file should have a single header line.
* Create the files `train.txt`, `val.txt`, and `test.txt` to specify the names of the files in each split.

### Citation ###

If you use or build upon this repository, please cite our paper:

J. Ventura, C. Pawlak, M. Honsberger, C. Gonsalves, J. Rice, N.L.R. Love, S. Han, V. Nguyen, K. Sugano, J. Doremus, G.A. Fricker, J. Yost, and M. Ritter (2024). [Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery.](https://www.sciencedirect.com/science/article/pii/S1569843224002024)  International Journal of Applied Earth Observation and Geoinformation, 130, 103848.

### Acknowledgments ###

This project was funded by the NASA Coastal Resilience grant (award number: 80NSSC23K0127). The research was also supported by the research grant from Department of Environmental Sciences and the support from the School of Data Science Capstone project at the University of Virginia. We give special thanks to Thomas Lever, Mahin Ganesan, Nicholas Miller, and Brendan Jalali for their contribution in modifying and testing the network in the early-stage.
