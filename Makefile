#### virtual environment ####

# all recipe lines for each target will be provided to a single invocation of the shell
.ONESHELL:

# make alone will run
.DEFAULT_GOAL := $(VENV)

SRC		= ./src/

# virtual environment, pip and python
VENV		= ./venv/
V_PIP		= ./venv/bin/pip
V_PY		= ./venv/bin/python
V_FLAKE		= ./venv/bin/flake8 

$(VENV): requirements.txt
	@echo "Installing Virtual Environment"
	pip install virtualenv
	virtualenv $(VENV)
	source ./venv/bin/activate
	$(V_PIP) install --upgrade -r requirements.txt

list:
	$(V_PY) -m pip list

flake:
	$(V_FLAKE) $(SRC)

run:
	$(V_PY) $(SRC)main.py

clean:
	@echo "Removing __pycache__ "
	find . -type d -name "__pycache__" | xargs rm -rf {};
	find -iname "*.pyc" -delete
	@echo "Removing virtual environment"
	rm -rf $(VENV)

.PHONY: list flake run clean

#ON:
#source ./venv/bin/activate
#OFF:
#deactivate