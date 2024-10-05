



echo [$(date)] : "Starting init_setup.sh"


# create a virtual environment

echo [$(date)] : "Creating virtual environment"

conda create --prefix ./env python=3.12 -y

echo [$(date)] : "Virtual environment created"

# Activate the virtual environment

echo [$(date)] : "Activating virtual environment"

#activate the virtual environment and return the status code of the activation


source activate ./env 

echo [$(date)] : "Virtual environment activated "

conda init && conda activate ./env &&  echo "Terminal activated" || echo "Terminal activation failed"

echo [$(date)] : "Virtual environment activated in terminal successfully"


echo [$(date)] : "Environment activated successfully"



# Install the required packages

echo [$(date)] : "Installing required packages"

pip install -r requirements.txt
# change the requirements_dev.txt to the requirements.txt if you are using cuda 

echo [$(date)] : "Packages installed"

echo [$(date)] : "END"


