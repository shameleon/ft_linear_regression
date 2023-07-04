run:
    python app.py

#init
setup:
    pip install -r requirements.txt

test:
    py.test tests

clean:
    rm -rf __pycache__

.PHONY: run setup test clean

#https://ricardoanderegg.com/posts/makefile-python-project-tricks/
#https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html