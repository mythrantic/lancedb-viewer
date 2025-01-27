FROM ubuntu:20.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required tools
RUN apt update && apt install -y \
    sudo \
    curl \
    git \
    software-properties-common \
    gnupg \
    lsb-release \
    apt-transport-https && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add a new user with passwordless sudo privileges
RUN useradd -m -s /bin/bash valiantlynx && echo "valiantlynx:valiantlynx" | chpasswd && \
    usermod -aG sudo valiantlynx && \
    echo "valiantlynx ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the new user
USER valiantlynx
WORKDIR /home/valiantlynx

# Install `uv` and ensure it's available in the PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc && \
    echo "source $HOME/.local/bin/env" >> ~/.bashrc && \
    export PATH=$HOME/.local/bin:$PATH && \
    uv --version

# Add the vault password as a build argument and save it to the vault.secret file
ARG VAULT_PASSWORD
RUN mkdir -p ~/.ansible-vault && echo "$VAULT_PASSWORD" > ~/.ansible-vault/vault.secret && chmod 600 ~/.ansible-vault/vault.secret

# Run the dotfiles installation script
RUN export USER=valiantlynx && bash -c "$(curl -fsSL https://raw.githubusercontent.com/valiantlynx/dotfiles/main/bin/dotfiles)" -- --tags bash,git,neovim,tmux,fzf,python

# Clean up passwordless sudo for security
USER root
RUN sed -i '/valiantlynx ALL=(ALL) NOPASSWD:ALL/d' /etc/sudoers

# Create a folder for the project and set permissions
RUN mkdir -p /home/valiantlynx/ai-alchemy && chown -R valiantlynx:valiantlynx /home/valiantlynx/ai-alchemy
WORKDIR /home/valiantlynx/ai-alchemy

# Switch back to valiantlynx user
USER valiantlynx

# Copy requirements and install dependencies using `uv`
COPY --chown=valiantlynx:valiantlynx ./requirements.txt ./
RUN bash -c "source ~/.bashrc"
# Copy the rest of the project
COPY --chown=valiantlynx:valiantlynx ./ ./

# Expose the necessary port
EXPOSE 8000

# Set the default command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

