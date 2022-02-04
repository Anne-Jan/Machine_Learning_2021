# Machine_Learning_2021
Repository for developing the semester project for the RUG course Machine Learning

The file `mnist_digits_data.txt` contains the small MNIST digits dataset used for this project.

The file `gridsearch_data_aug.py` was used to find the optimimal hyperparameters for the data augmentation. The file was used next`gridsearch_act_opt.py` for finding the optimal activation function and optimization algorithm for the CNN. 

The file `training_pipeline.py` contains the final training pipeline used for the final evaluation after gridsearch was performed.

All python files utilise the two functions in `augmentations.py` to perform the data augmentation.

All code can be run with the command `python3 <file.py>`. No other arguments are needed.
