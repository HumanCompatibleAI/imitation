.. _Ways To Contribute:

Ways to contribute
==================

There are four main ways you can contribute to imitation:

* :ref:`reporting-bugs`
* :ref:`suggesting-new-features`
* :ref:`contributing-to-the-documentation`
* :ref:`contributing-to-the-codebase`

Please note that by contributing to the project, you are agreeing to license your work under :ref:`imitation's MIT license <License>`, as per
`GitHub's terms of service <https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#6-contributions-under-repository-license>`_.


.. _`reporting-bugs`:

Reporting bugs
--------------

This section guides you through submitting a new bug report for imitation. Following the guidelines below helps maintainers and the community understand your report and reproduce the issue.

You can submit a new bug report by creating an issue on `GitHub <https://github.com/HumanCompatibleAI/imitation/issues/new>`_ and labeling it as a *bug*. **Before you do so, please make sure that**\ :


* You are using the `latest stable version <https://pypi.org/project/imitation/>`_ of imitation — to check your version, run ``pip show imitation``,
* You have read the relevant section of the `documentation <https://imitation.readthedocs.io/en/latest/>`_ that relates to your issue,
* You have checked `existing bug reports <https://github.com/HumanCompatibleAI/imitation/issues?q=is%3Aissue+label%3Abug+is%3Aopen>`_ to make sure that your issue has not already been reported, and
* You have a minimal, reproducible example of the issue.

When submitting a bug report, please **include the following information**\ :


* A clear, concise description of the bug,
* A minimal, reproducible example of the bug, with installation instructions, code, and error message,
* Information on your OS name and version, Python version, and other relevant information (e.g. hardware configuration if using the GPU), and
* Whether the problem arose when upgrading to a certain version of imitation, and if so, what version.

.. _`suggesting-new-features`:

Suggesting new features
-----------------------

This section explains how you can submit a new feature request, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your request and intended use cases and find related suggestions.

You can submit a new bug report by creating an issue on `GitHub <https://github.com/HumanCompatibleAI/imitation/issues/new>`_ and labeling it as an *enhancement*. **Before you do so, please make sure that**\ :


* You have checked the `documentation <https://imitation.readthedocs.io/en/latest/>`_ that relates to your request, as it may be that such feature is already available,
* You have checked `existing feature requests <https://github.com/HumanCompatibleAI/imitation/issues?q=is%3Aissue+label%3Aenhancement+is%3Aopen+>`_ to make sure that there is no similar request already under discussion, and
* You have a minimal use case that describes the relevance of the feature.

When you **submit the feature request**:


* Use a clear and descriptive title for the GitHub issue to easily identify the suggestion.
* Describe the current behavior, and explain what behavior you expected to see instead and why.
* If you want to request an API change, provide examples of how the feature would be used.
* If you want to request a new algorithm implementation, please provide a link to the relevant paper or publication.

.. _`contributing-to-the-documentation`:

Contributing to the documentation
---------------------------------

One of the simplest ways to start contributing to imitation is through improving the documentation. Currently, our documentation has some gaps, and we would love to have you help us fill them. You can help by adding missing sections of the API docs, editing existing content to make it more readable, clear and accessible, or contributing new content, such as tutorials and FAQs.

If you have struggled to understand something about our codebase and managed to figure it out in the end, please consider improving the relevant documentation section, or adding a tutorial or a FAQ entry, so that other users can learn from your experience.

Before submitting a pull request, please create an issue with the *documentation* label so that we can track the gap. You can then reference the issue in your pull request by including the issue number.

.. _`contributing-to-the-codebase`:

Contributing to the codebase
----------------------------

You can contribute to the codebase by proposing solutions to issues or feature suggestions you've raised yourself, or selecting an existing issue to work on. Please, make sure to create an issue on GitHub before you start working on a pull request, as explained in `Reporting bugs <#reporting-bugs>`_ and `Suggesting new features <#suggesting-new-features>`_.

Once you're ready to start working on your pull request, please make sure to follow our **coding style guidelines**\ :


* PEP8, with line width 88.
* Use the ``black`` autoformatter.
* Follow the `Google Python Style Guide <http://google.github.io/styleguide/pyguide.html>`_ unless
  it conflicts with the above. Examples of Google-style docstrings can be found
  `here <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

**Before you submit**\ , please make sure that:


* Your PR includes unit tests for any new features.
* Your PR includes type annotations, except when it would make the code significantly more complex.
* You have run the unit tests and there are no errors. We use ``pytest`` for unit testing: run ``pytest tests/`` to run the test suite.
* You should run ``pre-commit run`` to run linting and static type checks. We use ``pytype`` for static type analysis.

You may wish to configure this as a Git commit hook:

.. code-block:: bash

   pre-commit install

These checks are run on CircleCI and are required to pass before merging.
Additionally, we track test coverage by CodeCov and require that code coverage
should not decrease. This can be overridden by maintainers in exceptional cases.
Files in ``imitation/{examples,scripts}/`` have no coverage requirements.
