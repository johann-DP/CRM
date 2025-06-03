# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:0-3.12

# Installer les dépendances système
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/CRM

# 1) Copier la liste des dépendances et les installer 
COPY --chown=vscode:vscode ../requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copier l’intégralité du dépôt (sources, tests, setup.sh, etc.)
COPY --chown=vscode:vscode .. /workspaces/CRM

# Optionnel : installer pytest directement dans l’image
RUN pip install --no-cache-dir pytest
