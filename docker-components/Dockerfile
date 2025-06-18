FROM condaforge/mambaforge:latest

# Install system dependencies for ROCm 5.6, TA-Lib, Rust, and general use
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    software-properties-common \
    && wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.6 ubuntu main" > /etc/apt/sources.list.d/rocm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    libta-lib0 \
    rocm-libs=5.6 \
    rocm-dev=5.6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for PyO3
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH=/root/.cargo/bin:$PATH

# Set working directory
WORKDIR /app

# Copy environment.yml
COPY environment.yml .

# Create and configure Conda environment
RUN mamba env create -f environment.yml && \
    mamba clean --all --force-pkgs-dirs -y && \
    /opt/conda/envs/trading_env/bin/pip install \
        torch==2.2.2+rocm5.6 \
        torchaudio==2.2.2+rocm5.6 \
        torchvision==0.17.2+rocm5.6 \
        --index-url https://download.pytorch.org/whl/rocm5.6

# Activate environment
RUN echo "source activate trading_env" > ~/.bashrc
ENV PATH=/opt/conda/envs/trading_env/bin:$PATH

# Copy application code
COPY trading_dashboard.py backtestrocm3.py datalinux2.py ./

# Expose Dash app port
EXPOSE 8050

# Ensure GPU access for RX 7800 XT
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

CMD ["python", "trading_dashboard.py"]