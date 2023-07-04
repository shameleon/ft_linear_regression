PY      = python3
VENV    = venv
BIN     = $(VENV)/bin
SRC     = ./ft_linear_regression

# run:
# python $(PWD)/predict.py

# virtual environment
venv: $(VENV)

$(VENV): requirements.txt
    @echo 'Installing virtual environment'
    @virtualenv $(VENV)
    source $(BIN)/activate
    @pip install --upgrade -r requirements.txt

on: $(VENV)
    source env/bin/activate

off: $(VENV)
    source deactivate

test: $(VENV)
    $(BIN)/pytest
    py.test tests

flake: $(VENV)
    flake8 $(SRC)/*.py

list: $(VENV)
    $(PY) -m pip list

clean:
    @echo "Removing virtual environment"
    rm -rf $(VENV)
    rm -rf $(SRC)/__pycache__
    find . -type d -name "__pycache__" | xargs rm -rf {};
    find -iname "*.pyc" -delete

.PHONY: run venv on off test clean

# https://www.gnu.org/software/make/manual/html_node/index.html#SEC_Contents
# https://gist.github.com/genyrosk/50196ad03231093b604cdd205f7e5e0d
# https://venthur.de/2021-03-31-python-makefiles.html

# https://ricardoanderegg.com/posts/makefile-python-project-tricks/
# https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html