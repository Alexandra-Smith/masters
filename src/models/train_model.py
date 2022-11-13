import wandb

wandb.init(project="masters")

# Capture a dictionary of hyperparameters with config
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}

# Log metrics inside your training loop to visualize model performance
wandb.log({"loss": loss})

# Optional
wandb.watch(model)