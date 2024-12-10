MANAGE = pipenv run python manage.py

# step 1
shell:
	pipenv shell

# step 2
install:
	pipenv install --dev

# step 3
run:
	$(MANAGE) runserver

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .coverage
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.log' -delete