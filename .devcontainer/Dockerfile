# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:0-3.12

# 1) Installer les dépendances système
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Définir le dossier de travail
WORKDIR /workspaces/CRM

# 3) Copier requirements avant tout pour profiter du cache Docker
COPY --chown=vscode:vscode ../requirements.txt ./requirements.txt

# 4) Installer toutes les dépendances (dont pytest et le package lui-même)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest \
    && pip install --no-cache-dir -e .

# 5) Copier tout le reste du code dans l’image
COPY --chown=vscode:vscode .. /workspaces/CRM
