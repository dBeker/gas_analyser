
# Install python 

Follow this instruction: https://learnpython.com/blog/how-to-install-python-on-windows/

# How to run the code

Once you have python installed, open a new console (`windows + r and type cmd`) and navigate to the code's directory
(`cd /full/path/to/code/directory`)

Install required packages: 

`pip install -r requirements.txt`

This step has to be done only once. 

Now, you can execute the code

`python main.py`

If you wish to change any hyperparameter, check `parse_args` function in `main.py` and add the relevant parameter to the
launch. For example, if you want to edit leakage and membrane area, you can run the following command.

`python main.py -l 0.003 -ma 60`

