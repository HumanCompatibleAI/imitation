# Contributing

Please follow a coding style of:
  * PEP8, with line width 88.
  * Use the `black` autoformatter.
  * Follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html) unless
    it conflicts with the above. Examples of Google-style docstrings can be found
    [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

PRs should include unit tests for any new features, and add type annotations where possible. 
It is OK to omit annotations when it would make the code significantly more complex.

We use `pytest` for unit testing: run `pytest tests/` to run the test suite.
We use `pytype` for static type analysis.
You should run `ci/code_checks.sh` to run linting and static type checks,
and may wish to configure this as a Git commit hook:

```bash
ln -s ../../ci/code_checks.sh .git/hooks/pre-commit
```

These checks are run on CircleCI and are required to pass before merging.
Additionally, we track test coverage by CodeCov and require that code coverage
should not decrease. This can be overridden by maintainers in exceptional cases.
Files in `imitation/{examples,scripts}/` have no coverage requirements.
