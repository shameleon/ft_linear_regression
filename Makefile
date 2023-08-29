#############################
#### virtual environment ####
#############################

# all recipe lines for each target will be provided to a single invocation of the shell
# ONESHELL special target appears anywhere in the makefile then all recipe lines for each target will be provided to a single invocation of the shell.
.ONESHELL:

# make alone will run
.DEFAULT_GOAL := $(VENV)

SRC		= ./src_2/

# virtual environment, pip and python
VENV		= ./venv/
V_PIP		= ./venv/bin/pip
V_PY		= ./venv/bin/python
V_FLAKE		= ./venv/bin/flake8 

$(VENV): requirements.txt
	@echo "$(CG) Installing Virtual Environment $(CZ)"
	virtualenv $(VENV)
	@echo "$(CG) Virtual Environment pip is installing requirements $(CZ)"
	$(V_PIP) install -r requirements.txt

on:
	@echo "$(CG) Activating Virtual Environment $(CZ)"
	source ./venv/bin/activate

list:
	$(V_PY) -m pip list

flake:
	$(V_FLAKE) $(SRC)

run:
	$(V_PY) $(SRC)main.py

clean:
	@echo "$(CR) Removing __pycache__ "
	find . -type d -name "__pycache__" | xargs rm -rf {};
	@echo "$(CR) Removing .pyc files $(CZ)"
	find -iname "*.pyc" -delete

fclean: clean
	@echo "$(CR) Deactivating virtual environment"
	deactivate && rm -rf $(VENV)
	@echo "$(CR) Removed virtual environment $(CZ)"

re: fclean

.PHONY: list flake run clean fclean re on

# colors
CR:=\033[1;31m
CG:=\033[1;32m
CZ:=\033[0m

# ON:
# source ./venv/bin/activate
# OFF:
# deactivate