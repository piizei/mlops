## MLOPS demo

Idea is to demonstrate data science workflow where:
* User can work with local developer laptop to explore and model
* User can use any open source ML toolkit, in this example it is scikit-learn
* Model can be trained efficiently with remote (GPU) servers with same code
* Trained model can be deployed in continous fashion into as a rest endpoint


### Setup

* Create a new Machine Learning Service Workspace in portal.azure.com (or use a shared one)
* Download the configuration ('download config.json' from overview pane)
* This was developed with Python 3.7
* Download the MNIST dataset by running `python initial_setup`. In SR network you need to setup the proxies as env variables.


