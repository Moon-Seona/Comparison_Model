from model import CoNet

config = {
    "data_dir": "Clothing_and_Arts",
    "main_domain": "clothing", # main
    "aux_domain": "arts", # aux
    "lr": 0.001,
    "edim": 32, 
    "cross_layer": 2,
    "reg": 0.0001,
    "batch_size": 512,
    "std": 0.01
}
model = CoNet(config, print_summary=True)
model.fit()