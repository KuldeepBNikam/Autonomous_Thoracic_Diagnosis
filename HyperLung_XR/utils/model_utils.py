import os

def get_latest_model_path(artifact_dir="artifacts"):
    if not os.path.exists(artifact_dir):
        raise FileNotFoundError("Artifacts directory not found")

    timestamps = sorted(
        os.listdir(artifact_dir),
        reverse=True
    )

    for ts in timestamps:
        model_path = os.path.join(
            artifact_dir,
            ts,
            "model_training",
            "model.pt"
        )
        if os.path.exists(model_path):
            return model_path

    raise FileNotFoundError(
        "No trained model found. Run training pipeline first."
    )
