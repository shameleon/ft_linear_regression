#############################
#### virtual environment ####
#############################

# all recipe lines for each target will be provided to a single invocation of the shell
# ONESHELL special target appears anywhere in the makefile then all recipe lines for each target will be provided to a single invocation of the shell.
.ONESHELL:

# make alone will run
.DEFAULT_GOAL := $(VENV)

SRC		:= ./ft_linear_regression/

# virtual environment, pip and python
VENV		:= ./venv/
ACTIVATE	= $(VENV)bin/activate
V_PIP		= ./venv/bin/pip
V_PY		= ./venv/bin/python
V_FLAKE		= ./venv/bin/flake8 

# all: $(VENV)

$(ACTIVATE): requirements.txt
	@echo "$(CG) Installing Virtual Environment $(CZ)"
	virtualenv $(VENV)
	@echo "$(CG) venv pip is installing requirements $(CZ)"
	$(V_PIP) install -r requirements.txt

$(VENV): $(ACTIVATE)

#  list all the Python packages installed in an environment,
list:
	$(V_PY) -m pip list

flake:
	$(V_FLAKE) $(SRC)

run: $(VENV)
	$(V_PY) $(SRC)predict_price.py

clean:
	@echo "$(CR) Removing __pycache__ "
	find . -type d -name "__pycache__" | xargs rm -rf {};
	@echo "$(CR) Removing .pyc files $(CZ)"
	find -iname "*.pyc" -delete

fclean: clean
	@echo "$(CR) Removing virtual environment.$(CZ)"
	rm -rf $(VENV)

.PHONY: all list flake run clean fclean

# colors
CR:=\033[1;31m
CG:=\033[1;32m
CZ:=\033[0m

# ON:
# source ./venv/bin/activate
# OFF:
# deactivate