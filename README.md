IMPORTANT: The code works with the python modules: "numpy", "seaborn", "pandas", "matplotlib", "scikit-learn".

The project is structured as follows:

the main.ipynb file is a python notebook where I use the classes defined in ML_GRB.py to illustrate how one can implement it in a code;

the testing.ipynb file is a python notebook that I use to debug the code and test new functions before their official implementation;

the ML_GRB.py file is a python file where I define the classes "ML_GRB" (the main class) and "RND_Forest" (a model class that can be incorporate in the main class).
The idea is to create new model classes in the future that can be incorporate in ML_GRB just like RND_Forest does;

the functions.py file is a python file where I define all the functions that works as attributes for the class. 
For readability reason the functions are first defined in a separate file and then incorporated in the class file.

In the "data" folder there are the txt files of the data downloaded from the paper.

The images folder is there to collect important and relevant plot in the future. 