import torch.nn as nn
import torch
from chop.nn.modules import Identity
from pathlib import Path
import dill
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr
from chop.tools import get_tokenized_dataset
from chop.tools import get_trainer
import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from optuna.samplers import GridSampler, RandomSampler, TPESampler



checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"


dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [
        nn.Linear,
        Identity,
    ],
}



def construct_model(trial, is_grid):
    config = AutoConfig.from_pretrained(checkpoint)

    # Update the paramaters in the config
    for param in [
        "num_layers",
        "num_heads",
        "hidden_size",
        "intermediate_size",
    ]:
        if is_grid:
            chosen_val = trial.suggest_categorical(param, search_space[param])
            setattr(config, param, chosen_val) 
        else:
            chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
            setattr(config, param, search_space[param][chosen_idx])

    trial_model = AutoModelForSequenceClassification.from_config(config)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            if new_layer_cls == nn.Linear:
                continue
            elif new_layer_cls == Identity:
                new_layer = Identity()
                deepsetattr(trial_model, name, new_layer)
            else:
                raise ValueError(f"Unknown layer type: {new_layer_cls}")

    return trial_model


sampler_accuracy = {"RandomSampler":[], "TPESampler":[] , "GridSampler":[]}

def objective(trial, is_grid=False):

    # Define the model
    model = construct_model(trial, is_grid)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    sampler_accuracy[sampler_type].append(eval_results["eval_accuracy"]) 

    # Set the model as an attribute so we can fetch it later
    trial.set_user_attr("model", model)

    return eval_results["eval_accuracy"]


    
sampler_type = "RandomSampler"
sampler = RandomSampler()
print("RandomSampler")
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
print(sampler_accuracy)
best_so_far = max(sampler_accuracy[sampler_type])
model = study.best_trial.user_attrs["model"].cpu()
with open(f"{Path.home()}/adlsystems/tutorial_5_best_model.pkl", "wb") as f:
    dill.dump(model, f)

sampler_type = "TPESampler"
sampler = TPESampler()
print("TPESampler")
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
print(sampler_accuracy)
if max(sampler_accuracy[sampler_type])>best_so_far:
    best_so_far =  max(sampler_accuracy[sampler_type])
    model = study.best_trial.user_attrs["model"].cpu()
    with open(f"{Path.home()}/adlsystems/tutorial_5_best_model.pkl", "wb") as f:
        dill.dump(model, f)

sampler_type = "GridSampler"
grid_search_space = search_space.copy()
# Letting gridsampler know the layers beforehand
config = AutoConfig.from_pretrained(checkpoint)
trial_model = AutoModelForSequenceClassification.from_config(config)
for name, layer in trial_model.named_modules():
    if isinstance(layer, nn.Linear):
        grid_search_space[f"{name}_type"] = grid_search_space['linear_layer_choices']
sampler = GridSampler(grid_search_space)
print("GridSampler")
study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler,
)
study.optimize(
    lambda trial: objective(trial, is_grid=True),
    n_trials=100,
    timeout=60 * 60 * 24,
)
print(sampler_accuracy)
if max(sampler_accuracy[sampler_type])>best_so_far:
    best_so_far =  max(sampler_accuracy[sampler_type])
    model = study.best_trial.user_attrs["model"].cpu()
    with open(f"{Path.home()}/adlsystems/tutorial_5_best_model.pkl", "wb") as f:
        dill.dump(model, f)



# {'RandomSampler': [0.8506, 0.8544, 0.85656, 0.86068, 0.85324, 0.84168, 0.85992, 0.86644, 0.83412, 0.84496, 0.8488, 0.8278, 0.83996, 0.86068, 0.85596, 0.83484, 0.85128, 0.8644, 0.83232, 0.8468, 0.84156, 0.82756, 0.85016, 0.82072, 0.84928, 0.74756, 0.82536, 0.83412, 0.82552, 0.85672, 0.82872, 0.85808, 0.81516, 0.84252, 0.8346, 0.861, 0.78672, 0.86252, 0.85784, 0.84256, 0.85636, 0.82972, 0.8602, 0.84536, 0.83212, 0.86152, 0.83152, 0.86492, 0.84752, 0.83896, 0.85964, 0.85284, 0.84848, 0.8462, 0.82412, 0.85876, 0.84224, 0.83128, 0.86036, 0.84052, 0.83972, 0.85112, 0.82576, 0.85672, 0.83612, 0.84448, 0.84732, 0.85364, 0.83948, 0.8388, 0.86508, 0.8408, 0.86424, 0.854, 0.835, 0.85544, 0.84616, 0.86036, 0.85664, 0.86416, 0.8604, 0.86088, 0.86028, 0.86812, 0.83816, 0.83992, 0.8474, 0.86616, 0.85952, 0.85996, 0.86112, 0.8448, 0.8566, 0.85656, 0.85596, 0.86108, 0.82852, 0.8554, 0.83304, 0.85804],
# 'TPESampler': [0.84616, 0.83508, 0.82768, 0.838, 0.83096, 0.83888, 0.85916, 0.85712, 0.84396, 0.85328, 0.86392, 0.86392, 0.86392, 0.86392, 0.85828, 0.85656, 0.86392, 0.86124, 0.86288, 0.86032, 0.85972, 0.86392, 0.85828, 0.8578, 0.86124, 0.86392, 0.86124, 0.86288, 0.85876, 0.84188, 0.86392, 0.86392, 0.86392, 0.86124, 0.85536, 0.85448, 0.85988, 0.85984, 0.86016, 0.86176, 0.84868, 0.86392, 0.86392, 0.86392, 0.86232, 0.83092, 0.84636, 0.86392, 0.85984, 0.80652, 0.85896, 0.86392, 0.86392, 0.86392, 0.86392, 0.85548, 0.85148, 0.84092, 0.86764, 0.85896, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.86508, 0.86072, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.871, 0.85752, 0.85964, 0.84864, 0.86288, 0.871, 0.871, 0.871, 0.871, 0.871, 0.86288, 0.86916, 0.86916, 0.86824, 0.84664, 0.86824, 0.86824, 0.86824, 0.86824, 0.86488, 0.86404, 0.86824, 0.86488, 0.85792], 
#'GridSampler': [0.86552, 0.86292, 0.8622, 0.86128, 0.86216, 0.8542, 0.814, 0.85972, 0.83864, 0.85608, 0.83596, 0.85276, 0.86492, 0.85516, 0.85232, 0.84476, 0.84124, 0.85036, 0.86644, 0.83488, 0.83932, 0.8616, 0.82856, 0.86184, 0.85088, 0.85156, 0.84468, 0.8448, 0.85388, 0.85272, 0.862, 0.8332, 0.81924, 0.84868, 0.85272, 0.8258, 0.8526, 0.83952, 0.8548, 0.79824, 0.85812, 0.85736, 0.8454, 0.851, 0.8566, 0.84416, 0.8524, 0.85592, 0.82616, 0.84372, 0.84708, 0.81848, 0.8354, 0.86104, 0.83904, 0.86176, 0.859, 0.86568, 0.85416, 0.832, 0.83064, 0.86652, 0.83396, 0.8386, 0.84416, 0.83768, 0.85788, 0.8528, 0.842, 0.86112, 0.85604, 0.82788, 0.8476, 0.83832, 0.83956, 0.83124, 0.85576, 0.83004, 0.81892, 0.85936, 0.854, 0.83632, 0.84276, 0.84916, 0.83836, 0.85488, 0.8294, 0.83024, 0.84896, 0.8556, 0.83936, 0.84524, 0.85912, 0.84404, 0.84252, 0.85284, 0.8344, 0.85664, 0.8652, 0.8488]}


df = pd.DataFrame(sampler_accuracy)
df_cumulative = df.cummax()
df_cumulative["Trial"] = df.index + 1
csv_filename = "sampler_cumulative_results.csv"
df_cumulative.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")
df_melted = df_cumulative.melt(id_vars=["Trial"], var_name="Sampler", value_name="Best Accuracy So Far")


plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted, x="Trial", y="Best Accuracy So Far", hue="Sampler", linewidth=2, drawstyle="steps-post")
plt.xlabel("Trial Number", fontsize=12)
plt.ylabel("Best Accuracy", fontsize=12)
plt.title("Progression of Best Accuracy Over Trials", fontsize=14)
plt.grid(True)
plt.legend(title="Sampler")
plt.savefig("Sampler_Comparison.png")

# Find the maximum accuracy reached by each sampler
best_accuracies = df_cumulative.drop(columns=["Trial"]).max()
# Find the best sampler (one with the highest final accuracy)
best_sampler = best_accuracies.idxmax()
best_accuracy_value = best_accuracies.max()
print(f"Best Sampler: {best_sampler} with accuracy {best_accuracy_value:.4f}")

