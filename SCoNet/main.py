from model import CoNet

config = {
    "data_dir": "Clothing_and_Arts",
    "main_domain": "clothing", # main
    "aux_domain": "arts", # aux
    "lr": 0.00003,
    "edim": 10, 
    "cross_layer": 2,
    "reg": 0.0001,
    "batch_size": 1024,
    "std": 0.01,
    "epoch": 1000
}
model = CoNet(config, print_summary=True)
model.fit()