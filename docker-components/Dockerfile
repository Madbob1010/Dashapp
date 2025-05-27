FROM condaforge/mambaforge:latest

# Install minimal system dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	curl \
	git \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

# Set working directory

WORKDIR /app

# Install Python 3.10.16 and create conda environment with mamba

COPY environment.yml .

RUN mamba env create -f environment.yml && \
	mamba install -y -c pytorch pytorch-cpu && \
	mamba clean --all --force-pkgs-dirs -y

# Activate environment
RUN echo "source activate trading_env" > ~/.bashrc

ENV PATH /opt/conda/envs/trading_env/bin:$PATH

# COPY APPLICATION CODE

COPY trading_dashboard.py backtestrocm3.py datalinux2.py ./

# Expose Dash app port

EXPOSE 8050

CMD ["python", "trading_dashboard.py"]