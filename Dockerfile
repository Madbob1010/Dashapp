FROM continuumio/anaconda3:latest
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "trading_env", "/bin/bash", "-c"]
RUN conda install -c conda-forge pytorch-rocm
COPY . .
EXPOSE 8050
CMD ["conda", "run", "-n", "trading_env", "python", "trading_dashboard.py"]