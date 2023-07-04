PY		= python
VENV    = ./venv/
BIN     = $(VENV)bin
SRC     = ./ft_linear_regression/

# run:
# python $(PWD)/predict.py

# virtual environment
all: $(VENV)

$(VENV): requirements.txt
	virtualenv $(VENV)
	source $(BIN)/activate
	pip install --upgrade -r requirements.txt

on: $(VENV)
	source $(BIN)/activate

off: $(VENV)
	source deactivate

test: $(VENV)
	$(BIN)/pytest
	py.test tests

flake: $(VENV)
	flake8 $(SRC)*.py

list: $(VENV)
	$(PY) -m pip list

clean:
	@echo "Removing virtual environment"
	find . -type d -name "__pycache__" | xargs rm -rf {};
	find -iname "*.pyc" -delete

fclean: off clean
	rm -rf $(VENV)

.PHONY: all on off test flake list clean

# https://www.gnu.org/software/make/manual/html_node/index.html#SEC_Contents
# https://gist.github.com/genyrosk/50196ad03231093b604cdd205f7e5e0d
# https://venthur.de/2021-03-31-python-makefiles.html

# https://ricardoanderegg.com/posts/makefile-python-project-tricks/
# https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html