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