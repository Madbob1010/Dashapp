import torch, talib, vectorbt, plotly, dash
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(talib.__version__)
print(vectorbt.__version__)
print(plotly.__version__)
print(dash.__version__)
