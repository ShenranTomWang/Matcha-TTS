import wandb

WANDB_PROJECT = "MatchaTTS-HiFiGAN"
WANDB_NAME = "Multilingual Experiment A100 Balanced Dataset"
WANDB_DATASET = "multilingual-test"
WANDB_ARCH = "MatchaTTS: language embedding, HiFiGAN: vanilla, general"
device = "a100"

if __name__ == "__main__":
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_NAME,
        config={"architecture": WANDB_ARCH, "dataset": WANDB_DATASET, "hardware": device},
    )
    speaker_scores = {
        "AT/stoi": 0.03476608173971536,
        "AT/pesq": 1.3493842947483063,
        "AT/mcd": 17.68556912162006,
        "AT/f0_rmse": 75.88562759206604,
        "AT/las_rmse": 7.984762466948555,
        "AT/vuv_f1": 0.7256836658617088,
        "MJ/stoi": 0.038799910052540354,
        "MJ/pesq": 1.137168389558792,
        "MJ/mcd": 21.586509478161382,
        "MJ/f0_rmse": 136.81219217901196,
        "MJ/las_rmse": 7.5881106934620925,
        "MJ/vuv_f1": 0.6254860752899867,
        "JJ/stoi": 0.03870952746060171,
        "JJ/pesq": 1.2363521265983581,
        "JJ/mcd": 18.632428179269404,
        "JJ/f0_rmse": 58.00454544422187,
        "JJ/las_rmse": 4.929845884924541,
        "JJ/vuv_f1": 0.8333261497732738,
        "NJ/stoi": 0.04085142416153977,
        "NJ/pesq": 1.124665106534958,
        "NJ/mcd": 17.486743082985537,
        "NJ/f0_rmse": 89.88091947113182,
        "NJ/las_rmse": 6.35434139649528,
        "NJ/vuv_f1": 0.8145662591581504
    }
    general_scores = {
        "rtfs_mean": 0.030972,
        "rtfs_std": 0.095306,
        "rtfs_w_mean": 0.054327,
        "rtfs_w_std": 0.244068,
        "num_ode_steps": 10,
        "temperature": 0.667,
        "length_scale": 1.0
    }
    wandb.log(speaker_scores)
    wandb.log(general_scores)
