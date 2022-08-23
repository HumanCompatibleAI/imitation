SRC_FILES=src/ tests/ experiments/ examples/ docs/conf.py setup.py

typecheck:
	mypy ${SRC_FILES}
	pytype -j auto "${SRC_FILES}"


shellcheck:
	find . -path ./venv -prune -o -name '*.sh' -print0 | xargs -0 shellcheck

formatcheck:
	flake8 --darglint-ignore-regex '.*' "${SRC_FILES[@]}"
	black --check --diff "${SRC_FILES[@]}"
	codespell -I .codespell.skip --skip='*.pyc,tests/testdata/*,*.ipynb,*.csv' "${SRC_FILES[@]}"

docscheck:
	pushd docs/
	make clean
	make html
	popd