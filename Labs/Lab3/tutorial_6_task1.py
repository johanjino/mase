import torch
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearLog,
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearBlockLog,
    LinearBinary,
    LinearBinaryScaling,
    LinearBinaryResidualSign,
)
from transformers import AutoModel
from pathlib import Path
import dill
from chop.tools import get_tokenized_dataset
from chop.tools.utils import deepsetattr
from copy import deepcopy
from chop.tools import get_trainer
import random
import optuna


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from optuna.samplers import GridSampler, RandomSampler, TPESampler


checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

with open(f"{Path.home()}/adlsystems/tutorial_5_best_model.pkl", "rb") as f:
    base_model = dill.load(f)


dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

search_space = {
    "linear_layer_choices": [
        torch.nn.Linear,
        LinearInteger,
    ],
}


def construct_model(trial):

    # Fetch the model
    trial_model = deepcopy(base_model)

    # Quantize layers according to optuna suggestions
    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            if new_layer_cls == torch.nn.Linear:
                continue

            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }

            # If the chosen layer is integer, define the low precision config
            if new_layer_cls == LinearInteger:
                bits_width = [8, 16, 32] 
                frac_width = [2, 4, 8]
                chosen_width = bits_width[trial.suggest_int("bits_width", 0, 2)]
                chosen_frac = frac_width[trial.suggest_int("frac_width", 0, 2)]
                kwargs["config"] = {
                    "data_in_width": chosen_width,
                    "data_in_frac_width": chosen_frac,
                    "weight_width": chosen_width,
                    "weight_frac_width": chosen_frac,
                    "bias_width": chosen_width,
                    "bias_frac_width": chosen_frac,
                }
            # elif... (other precisions)

            # Create the new layer (copy the weights)
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data = layer.weight.data

            # Replace the layer in the model
            deepsetattr(trial_model, name, new_layer)

    return trial_model

accuracy_per_trial = []


def objective(trial):

    # Define the model
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    trial.set_user_attr("model", model)

    accuracy_per_trial.append(eval_results["eval_accuracy"])

    return eval_results["eval_accuracy"]


sampler = RandomSampler()


study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler,
)

study.optimize(
    objective,
    n_trials=100,
    timeout=60 * 60 * 24,
)



# Save to CSV
df = pd.DataFrame({"Trial": np.arange(1, len(accuracy_per_trial) + 1), "Accuracy": accuracy_per_trial})
df.to_csv("tutorial_6_accuracy_per_trial.csv", index=False)

# Compute max accuracy so far
max_so_far = np.maximum.accumulate(accuracy_per_trial)

# Plot using Seaborn
plt.figure(figsize=(10, 5))
sns.lineplot(x=df["Trial"], y=max_so_far, label="Max Accuracy So Far", color="b")

plt.xlabel("Trial")
plt.ylabel("Max Accuracy")
plt.title("Max Accuracy Over Trials")
plt.legend()
plt.savefig("MixedPrecision_task1.png")