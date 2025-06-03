# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:0-3.12

# Installe les dépendances système si besoin
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /workspaces/CRM

# Copier et installer les dépendances Python
COPY --chown=vscode:vscode ../requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet dans le conteneur
COPY --chown=vscode:vscode .. /workspaces/CRM
