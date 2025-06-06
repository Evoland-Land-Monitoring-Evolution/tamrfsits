all: check test


#######################
# Testing and linting #
#######################
check: test lint

test-perf:
	pytest -m "performances" tests/

test-smoke:
	pytest -m "not requires_data and not requires_gpu and not performances" tests/

test-all:
	pytest -m "not performances" tests/

test-hydra-smoke:
	pytest -m "not requires_data and not requires_gpu and check_hydra" tests/

test-hydra-all:
	pytest -m "check_hydra" tests/

PYLINT_IGNORED = ""

pylint:
	echo "\npylint:\n======="  && pylint --ignore-patterns "flycheck_*" --ignore=$(PYLINT_IGNORED) --load-plugins=perflint src/ tests/ bin/

pylint-core:
	echo "\npylint:\n======="  && pylint --ignore-patterns "flycheck_*" --ignore=$(PYLINT_IGNORED) --load-plugins=perflint src/


ruff:
	echo "\nruff:\n=====" && ruff check src/ bin/ tests/ --output-format pylint


rufffix:
	echo "\nruff:\n=====" && ruff check src/ bin/ tests/ --fix --output-format pylint


mypy:
	echo "\nmypy:\n=====" && mypy --exclude flycheck* src/ tests/ bin/


pyupgrade:
	echo "\npyupgrade:\n========="
	find ./src/ -type f -name "*.py" -print |xargs pyupgrade --py310-plus
	find ./tests/ -type f -name "*.py" -print |xargs pyupgrade --py310-plus
	find ./bin/ -type f -name "*.py" -print |xargs pyupgrade --py310-plus


autowalrus:
	echo "\nautowalrus:\n==========="
	find ./src/ -type f -name "*.py" -print |xargs auto-walrus
	find ./tests/ -type f -name "*.py" -print |xargs auto-walrus
	find ./bin/ -type f -name "*.py" -print |xargs auto-walrus

isort:
	echo "\nisort:\n==========="
	find src/ -type f -name "*.py" -print |xargs isort
	find tests/ -type f -name "*.py" -print |xargs isort
	find bin/ -type f -name "*.py" -print |xargs isort


refurb:
	echo "\nrefurb:\n======="
	refurb tests/
	refurb bin/
	cd src/ && refurb tamrfsits/

precommit:
	echo "\nprecommit:\n======="
	pre-commit


lint: pylint ruff mypy refurb

fix: isort rufffix pyupgrade autowalrus
