# GAN-enabled metasuraface design

## Requirements

We recommend using python3 and a virtual environment

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

## Training the GAN

The training set is in `Data` folder. `trainset.nc` comprises of 500 high efficiency deflectors with size 64 x 256. They are only the half of a full device (128 x 256), because we enforce the reflection symmetry along y direction. You can see more details of the training set by loading them with python or matlab.

You can change the parameters by editing `Params.json` in `Result` folder. 

If you want to train the network, simply run
```
python main.py 
```

or 

```
python main.py --output_dir Result --train_path Data/trainset.nc
```

to specify non-default output folder or training set


## Results

All results will store in output_dir folder.
```
-figures/  (figures of generated devices and loss function curve for every 250 iterations)
-model/    (all weights of the generator and discriminator)
-outputs/  (500 generated devices for every combination of wavelength and angle in `.mat` format)
```

## Citation
If you use this code for your research, please cite:

[Free-form diffractive metagrating design based on generative adversarial networks.<br>](https://fanlab.stanford.edu/wp-content/papercite-data/pdf/jiang2019free.pdf)
J. Jiang, D. Sell, S. Hoyer, J. Hickey, J. Yang, and J. A. Fan


