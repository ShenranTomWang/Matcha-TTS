import wandb

WANDB_PROJECT = "MatchaTTS-Vocos"
WANDB_NAME = "Multilingual Experiment A100"
WANDB_DATASET = "multilingual-test"
WANDB_ARCH = "MatchaTTS: language embedding, Vocos: vanilla"
device = "a100"

if __name__ == "__main__":
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_NAME,
        config={"architecture": WANDB_ARCH, "dataset": WANDB_DATASET, "hardware": device},
    )
    scores = {
        "AT/stoi": 0.034467383276918964,
        "AT/pesq": 1.363548581600189,
        "AT/mcd": 16.88833688779086,
        "AT/f0_rmse": 74.89061267037675,
        "AT/las_rmse": 7.055516948463254,
        "AT/vuv_f1": 0.7355746955585164,
        "MJ/stoi": 0.03716078347668129,
        "MJ/pesq": 1.1734014856815338,
        "MJ/mcd": 21.607332695152845,
        "MJ/f0_rmse": 136.14104101854554,
        "MJ/las_rmse": 7.237321248480112,
        "MJ/vuv_f1": 0.6754111697184492,
        "JJ/stoi": 0.03586410085988776,
        "JJ/pesq": 1.3102074790000915,
        "JJ/mcd": 18.78067436443153,
        "JJ/f0_rmse": 57.59042086985218,
        "JJ/las_rmse": 5.256278871686704,
        "JJ/vuv_f1": 0.8345648743704168,
        "NJ/stoi": 0.03309164618486104,
        "NJ/pesq": 1.1419564771652222,
        "NJ/mcd": 17.36092230950615,
        "NJ/f0_rmse": 87.35175754309842,
        "NJ/las_rmse": 6.174249493614179,
        "NJ/vuv_f1": 0.8293946476192314,
        "rtfs_mean": 0.038103,
        "rtfs_std": 0.234020,
        "rtfs_w_mean": 0.061516,
        "rtfs_w_std": 0.433657,
        "num_ode_steps": 10,
        "temperature": 0.667,
        "length_scale": 1.0,
    }
    wandb.log(scores)
    