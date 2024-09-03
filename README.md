# Computer-Vision-with-Pytorch

**core** folder contains modules that are crucial for the framework such as datasets, loss_fuctions and optimizers, plotting, etc

**data** folder is where all the downloaded datasets are stored

**trained_model_storaged** folder is where the framework stores the trained models information (hundreds and thousands of weight and bias values)

**util** folder contains functions that are used but are not crucial such as changing time format from [seconds] to [hours:minutes:seconds] 

**train, test, continued_train** are the files needed for training, testing, and additionally training a certain model which are all used in the main file.

The whole process of deciding the model, training parameters, and then training and testing it can be found in the **main** file
