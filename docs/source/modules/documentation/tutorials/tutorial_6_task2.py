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
    # LinearBinaryResidualSign,
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

import sys

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

search_space_options = {
    0: {"linear_layer_choices": [torch.nn.Linear, LinearInteger]},
    1: {"linear_layer_choices": [torch.nn.Linear, LinearMinifloatDenorm]},
    2: {"linear_layer_choices": [torch.nn.Linear, LinearMinifloatIEEE]},
    3: {"linear_layer_choices": [torch.nn.Linear, LinearLog]},
    4: {"linear_layer_choices": [torch.nn.Linear, LinearBlockFP]},
    # 5: {"linear_layer_choices": [torch.nn.Linear, LinearBlockMinifloat]},
    6: {"linear_layer_choices": [torch.nn.Linear, LinearBlockLog]},
    7: {"linear_layer_choices": [torch.nn.Linear, LinearBinary]},
    8: {"linear_layer_choices": [torch.nn.Linear, LinearBinaryScaling]},
    9: {
        "linear_layer_choices": [
            torch.nn.Linear,
            LinearInteger,
            LinearMinifloatDenorm,
            LinearMinifloatIEEE,
            LinearLog,
            LinearBlockFP,
            # LinearBlockMinifloat,
            LinearBlockLog,
            LinearBinary,
            LinearBinaryScaling,
        ]
    },
}


# search_space_options = {
#     0: {"linear_layer_choices": [LinearInteger]}, #good
#     1: {"linear_layer_choices": [LinearMinifloatDenorm]}, #good
#     2: {"linear_layer_choices": [LinearMinifloatIEEE]}, #good
#     3: {"linear_layer_choices": [LinearLog]}, #good
#     4: {"linear_layer_choices": [LinearBlockFP]}, #good
#     5: {"linear_layer_choices": [LinearBlockMinifloat]},
#     6: {"linear_layer_choices": [LinearBlockLog]}, #good
#     7: {"linear_layer_choices": [LinearBinary]}, #good
#     8: {"linear_layer_choices": [LinearBinaryScaling]}, #good
#     9: {
#         "linear_layer_choices": [
#             torch.nn.Linear,
#             LinearInteger,
#             LinearMinifloatDenorm,
#             LinearMinifloatIEEE,
#             LinearLog,
#             LinearBlockFP,
#             LinearBlockMinifloat,
#             LinearBlockLog,
#             LinearBinary,
#             LinearBinaryScaling,
#         ]
#     },
# }
# execution_type = 8

execution_type = int(sys.argv[1])
search_space = search_space_options.get(execution_type, {})
print(search_space)
if search_space["linear_layer_choices"] == {}:
    raise Exception("Invalid option chosen... ")


def construct_model(trial):

    # Fetch the model
    trial_model = deepcopy(base_model)
    print(search_space["linear_layer_choices"])

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
                chosen_width = bits_width[trial.suggest_int("int_bits_width", 0, 2)]
                chosen_frac = frac_width[trial.suggest_int("int_frac_width", 0, 2)]
                kwargs["config"] = {
                    "data_in_width": chosen_width,
                    "data_in_frac_width": chosen_frac,
                    "weight_width": chosen_width,
                    "weight_frac_width": chosen_frac,
                    "bias_width": chosen_width,
                    "bias_frac_width": chosen_frac,
                }
            elif new_layer_cls in [LinearMinifloatIEEE, LinearMinifloatDenorm]:
                bits_width = [8, 16, 32] 
                exponent_width = [2, 4, 8]
                chosen_width = bits_width[trial.suggest_int("minifloat_bits_width", 0, 2)]
                chosen_exponent = exponent_width[trial.suggest_int("minifloat_exponent_width", 0, 2)]

                chosen_bias = None
                kwargs["config"] = {
                    "data_in_width": chosen_width,
                    "data_in_exponent_width": chosen_exponent,
                    "data_in_exponent_bias": chosen_bias,
                    "weight_width": chosen_width,
                    "weight_exponent_width": chosen_exponent,
                    "weight_exponent_bias": chosen_bias,
                    "bias_width": chosen_width,
                    "bias_exponent_width": chosen_exponent,
                    "bias_exponent_bias": chosen_bias,
                }
            elif new_layer_cls == LinearLog:
                bits_width = [8, 16, 32] 
                bias_width = [-1, 0, 1]
                chosen_width = bits_width[trial.suggest_int("log_bits_width", 0, 2)]
                chosen_bias = bias_width[trial.suggest_int("log_bias_width", 0, 2)]
                kwargs["config"] = {
                    "data_in_width": chosen_width,
                    "data_in_exponent_bias": chosen_bias,
                    "weight_width": chosen_width,
                    "weight_exponent_bias": chosen_bias,
                    "bias_width": chosen_width,
                    "bias_exponent_bias": chosen_bias,
                }
            elif new_layer_cls == LinearBlockFP:
                bits_width = [4, 8, 16] # Because exponent width isnt included in the bit width here
                exponent_width = [4, 8, 16]
                block_size = [8, 16, 32]
                chosen_width = bits_width[trial.suggest_int("blockfp_bits_width", 0, 2)]
                chosen_exponent = exponent_width[trial.suggest_int("blockfp_exponent_width", 0, 2)]
                chosen_block = block_size[trial.suggest_int("blockfp_block_width", 0, 2)]

                chosen_exponent_bias = None
                kwargs["config"] = {
                    "data_in_width": chosen_width,
                    "data_in_exponent_width": chosen_exponent,
                    "data_in_exponent_bias": chosen_exponent_bias,
                    "data_in_block_size": chosen_block,
                    "weight_width": chosen_width,
                    "weight_exponent_width": chosen_exponent,
                    "weight_exponent_bias": chosen_exponent_bias,
                    "weight_block_size": chosen_block,
                    "bias_width": chosen_width,
                    "bias_exponent_width": chosen_exponent,
                    "bias_exponent_bias": chosen_exponent_bias,
                    "bias_block_size": chosen_block,
                }
            elif new_layer_cls == LinearBlockMinifloat:
                bits_width = [8, 16, 32] # Total width
                block_size = [[8], [16], [32]]
                shared_exponent_bias = [2, 4, 8]
                chosen_width = bits_width[trial.suggest_int("block_minifp_bits_width", 0, 2)]
                chosen_block = block_size[trial.suggest_int("block_minifp_block_width", 0, 2)]
                chosen_exponent_bias = shared_exponent_bias[trial.suggest_int("block_minifp_shared_exponent_bias", 0, 2)]

                chosen_exponent = chosen_width/2
                kwargs["config"] = {
                    "data_in_width": chosen_width,
                    "data_in_exponent_width": chosen_exponent,
                    "data_in_exponent_bias_width": chosen_exponent_bias,
                    "data_in_block_size": chosen_block,
                    "weight_width": chosen_width,
                    "weight_exponent_width": chosen_exponent,
                    "weight_exponent_bias_width": chosen_exponent_bias,
                    "weight_block_size": chosen_block,
                    "bias_width": chosen_width,
                    "bias_exponent_width": chosen_exponent,
                    "bias_exponent_bias_width": chosen_exponent_bias,
                    "bias_block_size": chosen_block,
                }
            elif new_layer_cls == LinearBlockLog:
                bits_width = [8, 16, 32] 
                block_size = [[8], [16], [32]]
                shared_exponent_bias = [2, 4, 8]
                chosen_width = bits_width[trial.suggest_int("block_log_bits_width", 0, 2)]
                chosen_block = block_size[trial.suggest_int("block_log_width", 0, 2)]
                chosen_exponent_bias = shared_exponent_bias[trial.suggest_int("block_log_shared_exponent_bias", 0, 2)]

                kwargs["config"] = {
                    "data_in_width": chosen_width,
                    "data_in_exponent_bias_width": chosen_exponent_bias,
                    "data_in_block_size": chosen_block,
                    "weight_width": chosen_width,
                    "weight_exponent_bias_width": chosen_exponent_bias,
                    "weight_block_size": chosen_block,
                    "bias_width": chosen_width,
                    "bias_exponent_bias_width": chosen_exponent_bias,
                    "bias_block_size": chosen_block,
                }
            elif new_layer_cls == LinearBinary:
                stochastic = bool([trial.suggest_int("bin_stochastic", 0, 1)])
                bipolar = True
                kwargs["config"] = {
                    "weight_stochastic": stochastic,
                    "weight_bipolar": bipolar,
                }
            elif new_layer_cls == LinearBinaryScaling:
                stochastic = bool([trial.suggest_int("bin_scaling_stochastic", 0, 1)])
                bipolar = True
                
                binary_training = True
                # Should we be trying scaling just few?
                kwargs["config"] = {
                    "data_in_stochastic": stochastic,
                    "data_in_bipolar": bipolar,
                    "weight_stochastic": stochastic,
                    "weight_bipolar": bipolar,
                    "bias_stochastic": stochastic,
                    "bias_bipolar": bipolar,
                    "binary_training": binary_training,
                }
            elif new_layer_cls == LinearBinaryResidualSign:
                raise NotImplementedError
                

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


sampler = TPESampler()

study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler,
)

study.optimize(
    objective,
    n_trials=100,
    timeout=60 * 60 * 48,
)

print(execution_type, accuracy_per_trial)

if len(search_space["linear_layer_choices"]) > 2:
    csv_filename = "tut6_mixed_search.csv"
elif search_space["linear_layer_choices"] != {}:
    layer_name = search_space["linear_layer_choices"][-1].__name__
    layer_name = layer_name.lstrip("Linear")
    csv_filename = f"tut6_{layer_name}_search.csv"
else:
    csv_filename = f"tut6_unknown_search_{execution_type}.csv"

# Save to CSV
df = pd.DataFrame({"Trial": np.arange(1, len(accuracy_per_trial) + 1), "Accuracy": accuracy_per_trial})
df.to_csv(csv_filename, index=False)

# # Compute max accuracy so far
# max_so_far = np.maximum.accumulate(accuracy_per_trial)

# # Plot using Seaborn
# plt.figure(figsize=(10, 5))
# sns.lineplot(x=df["Trial"], y=max_so_far, label="Max Accuracy So Far", color="b")

# plt.xlabel("Trial")
# plt.ylabel("Max Accuracy")
# plt.title("Max Accuracy Over Trials")
# plt.savefig("Full_mixed_precision_search.png")