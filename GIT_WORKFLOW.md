# Git workflow

1. Create a feature branch before making project changes.
2. Keep source code, notebooks, configuration, workflow files, and documentation in commits.
3. Review generated reports and data separately before adding them; do not commit credentials or the local MLflow database.
4. Run python -m unittest discover -s tests and the Lab 8 workflow locally before opening a pull request.
5. Merge only after the required GitHub Actions checks pass.
