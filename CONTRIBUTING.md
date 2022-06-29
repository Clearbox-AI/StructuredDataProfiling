# Contributing

## Setting up the environment


### Using pre-commit hooks
[Pre-commit](https://pre-commit.com/) is a framework used to manage git hooks and we use it to deliver choerent linting and static checks to all the contributors.
[After you install pre-commit on your machine](https://pre-commit.com/#installation) you can enable the hooks for Structured Data Profiling by running
```sh
pre-commit install
```
from the repo root folder. From that point on, every time you do a `git commit` linters will run against your changes.
Hooks will auto-fix your changes. Note that if any hook fail, the commit will be blocked; you need to add the modified file and commit it again.
