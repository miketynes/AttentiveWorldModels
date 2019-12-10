# Attentive World Models


## Installation

Create a new conda env using our included yaml file. 

`conda env create -n AWM -f AWM_environment.yml`

This will create a new conda env entitled AWM. 

All of our code must be run in this environment.
If you want to run your notebooks, launch jupyter from this environment. 

**IF** you want to run the evolutionary compuatation (which takes forever) you will need to run: 
```bash
conda activate AWM
git clone git@github.com:hardmaru/estool.git
cd estool
pip install -e .```


## Demo
To show our demo agent, run `python TESTING_SCRIPT.py` 
This will run a rollout with our best world model and save the rollout in `./sad_car_noises.gif`
If you are running this on a remote server, you will need to enable X11 forwarding in your ssh session, which you can do with `ssh -XC user@host` when starting your session. 

## Training logs
All of our training was done in ipython notebooks. Our results are in the report, but if you want to look more closely, see: 

    * training_vae.ipynb
    * mdn-rnn.ipynb
    * attentive-mdn-rnn.ipynb
    * Controller.ipynb
    
Note: If you clear out one of these notebooks, you can find archived html versions in `./images/`
    
Some model code can be found in `models.py`.

Auxillary notebooks are `Collecting State.ipynb` for collecting our full dataset, and `Compress States.ipynb` for saving compressed (latent space) images after VAE training. 

Model weights are in `./weights/2019.12.07/`

