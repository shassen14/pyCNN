#!/usr/bin/env bash
# virtual environment name. Change name if so desired
VIRTUAL_ENVIRONMENT_DIR="venv"

# check for pip and install if not installed
REQUIRED_PIP="python3-pip"
PIP_OK=$(dpkg-query -W -f='${Status}\n' $REQUIRED_PIP | grep 'install ok installed')
echo -n "Checking for $REQUIRED_PIP: "
if [ "" = "$PIP_OK" ]; then
  echo "NOT installed. Setting up $REQUIRED_PIP."
  sudo apt --yes install $REQUIRED_PIP
else
  echo "Installed"
fi

# check for pip package virtualenv and install if not installed
REQUIRED_VENV="virtualenv"
VENV_OK=$(pip list | grep $REQUIRED_VENV)
echo -n "Checking for $REQUIRED_VENV: "
if [ "" = "$VENV_OK" ]; then
  echo "NOT installed. Setting up $REQUIRED_VENV."
  pip install $REQUIRED_VENV
else
  echo "Installed"
fi

# check for virtual environment directory and creates one if need be
echo -n "Virtual environment directory, $VIRTUAL_ENVIRONMENT_DIR, does "
if [ -d "$VIRTUAL_ENVIRONMENT_DIR" ]; then
  echo "exist."
else
  echo "NOT exist. Creating one."
  python3 -m venv ./$VIRTUAL_ENVIRONMENT_DIR
fi

# enter virtual environment
. $VIRTUAL_ENVIRONMENT_DIR/bin/activate

# TODO: could just check if the packages in requirements.txt are already installed
# and install them if not. this is just simpler to do
# install requirements if new
pip3 install -r requirements.txt
