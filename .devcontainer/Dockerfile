FROM mcr.microsoft.com/devcontainers/python:0-3.12

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/CRM

# Copier requirements et installer
COPY --chown=vscode:vscode ../requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet (dont setup.sh, setup.py, pred/, tests/, etc.)
COPY --chown=vscode:vscode .. /workspaces/CRM
