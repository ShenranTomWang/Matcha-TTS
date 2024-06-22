import wandb

WANDB_PROJECT = "MatchaTTS-Vocos"
WANDB_NAME = "Multilingual Experiment"
WANDB_DATASET = "multilingual-test"
WANDB_ARCH = "MatchaTTS: language embedding, Vocos: vanilla"
device = "a100"

if __name__ == "__main__":
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_NAME,
        config={
            "architecture": WANDB_ARCH,
            "dataset": WANDB_DATASET,
            "hardware": device
        }
    )
    scores = {
        "stoi": 0.03440895149910613,
        "pesq": 1.2474755695462227,
        "mcd": 18.674093612108255,
        "f0-rmse": 90.00115644060372,
        "las-rmse": 6.430459099262706,
        "vuv_f1": 0.7672153208009433,
        "num_ode_steps": 10,
        "rtfs_mean": 0.027102,
        "rtfs_std": 0.057015,
        "rtfs_w_mean": 0.039801,
        "rtfs_w_std": 0.060574
    }
    wandb.log(scores)
    
    