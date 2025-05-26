FROM continuumio/anaconda3:latest

WORKDIR /app

COPY . .
COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean --all -y && \
    echo "source activate trading_env" >> ~/.bashrc

RUN conda run -n trading_env pip install torch --index-url https://download.pytorch.org/whl/rocm5.6

ENV PATH=/opt/conda/envs/trading_env/bin:$PATH

CMD ["bash"]