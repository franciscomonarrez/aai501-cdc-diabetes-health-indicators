# Contributing

Branches

* main: stable
* dev: integration
* feature branches: feat/<short-name>

Workflow

1. Branch from dev
2. Code with tests or notes
3. Run `pre-commit run -a`
4. Open PR into dev
5. Periodically merge dev into main

Style

* PEP 8 via black and flake8
* Keep notebooks small and clear outputs before commit
* Shared code in src/aai501_diabetes
