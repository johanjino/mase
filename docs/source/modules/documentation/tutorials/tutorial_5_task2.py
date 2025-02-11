import torch.nn as nn
import torch
from chop.nn.modules import Identity
from pathlib import Path
import dill
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr
from chop.tools import get_tokenized_dataset
from chop.tools import get_trainer
from chop.pipelines import CompressionPipeline
import chop.passes as passes
from chop import MaseGraph
import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time


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



def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)

    # Update the paramaters in the config
    for param in [
        "num_layers",
        "num_heads",
        "hidden_size",
        "intermediate_size",
    ]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][chosen_idx])

    trial_model = AutoModelForSequenceClassification.from_config(config)
    trial_model.config.problem_type = "single_label_classification"

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


def gen_quantization_config():
    return {
        "by": "type",
        "default": {
            "config": {
                "name": None,
            }
        },
        "linear": {
            "config": {
                "name": "integer",
                # data
                "data_in_width": 8,
                "data_in_frac_width": 4,
                # weight
                "weight_width": 8,
                "weight_frac_width": 4,
                # bias
                "bias_width": 8,
                "bias_frac_width": 4,
            }
        },
    }

def gen_pruning_config():
    return {
        "weight": {
            "sparsity": 0.5,
            "method": "l1-norm",
            "scope": "local",
        },
        "activation": {
            "sparsity": 0.5,
            "method": "l1-norm",
            "scope": "local",
        },
    }


strategy_accuracy = {"WithoutComp":[], "CompAware":[] , "PostCompTraining":[]}

def objective(trial):
    # Define the model
    model = construct_model(trial)

    mg = MaseGraph(
        model,
        hf_input_names=[
            "input_ids",
            "attention_mask",
            "labels"
        ]
    )

    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(mg)

    if strategy_type == "WithoutComp":

        # Training for 2 epoch
        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=2,
        )

        trainer.train()

        # Evaluation
        eval_results = trainer.evaluate()

        strategy_accuracy[strategy_type].append(eval_results["eval_accuracy"]) 

        # Set the model as an attribute so we can fetch it later
        trial.set_user_attr("model", model)

        return eval_results["eval_accuracy"]

    elif strategy_type == "CompAware":

        # Training for 2 epoch
        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=2,
        )

        trainer.train()

        # Post traing compression
        # mg = MaseGraph(model.cpu(),
        #             hf_input_names=[
        #                 "input_ids",
        #                 "attention_mask",
        #                 "labels"
        #             ])

        mg.model = mg.model.cpu()
        pipe = CompressionPipeline()

        mg, _ = pipe(
            mg,
            pass_args={
                "quantize_transform_pass": gen_quantization_config(),
                "prune_transform_pass": gen_pruning_config(),
            },
        )

        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy"
        )

        # Evaluation
        eval_results = trainer.evaluate()

        strategy_accuracy["CompAware"].append(eval_results["eval_accuracy"]) 

        # Set the model as an attribute so we can fetch it later
        trial.set_user_attr("model", model)

        return eval_results["eval_accuracy"]
    
    elif strategy_type == "PostCompTraining":

        # Initial training 1 epoch
        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=1,
        )

        trainer.train()

        # Compression: Quantization and pruning
        # mg = MaseGraph(model.cpu(),
        #             hf_input_names=[
        #                 "input_ids",
        #                 "attention_mask",
        #                 "labels"
        #             ])
        mg.model = mg.model.cpu()
        pipe = CompressionPipeline()

        mg, _ = pipe(
            mg,
            pass_args={
                "quantize_transform_pass": gen_quantization_config(),
                "prune_transform_pass": gen_pruning_config(),
            },
        )

        # Post Compression training 1 epoch
        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=1,
        )

        trainer.train()

        # Evaluation
        eval_results = trainer.evaluate()

        strategy_accuracy["PostCompTraining"].append(eval_results["eval_accuracy"]) 

        # Set the model as an attribute so we can fetch it later
        trial.set_user_attr("model", model)

        return eval_results["eval_accuracy"]




strategy_type = "WithoutComp"
sampler = RandomSampler()
print("WithoutComp")
study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler,
)
start_time = time.time()
study.optimize(
    objective,
    n_trials=100,
    timeout=60 * 60 * 24,
)
end_time = time.time()
elapsed_time_1 = end_time - start_time
print(f"Total Search Time: {elapsed_time_1:.2f} seconds")
print(strategy_accuracy)

strategy_type = "CompAware"
sampler = RandomSampler()
print("CompAware")
study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler,
)
start_time = time.time()
study.optimize(
    objective,
    n_trials=100,
    timeout=60 * 60 * 24,
)
end_time = time.time()
elapsed_time_2 = end_time - start_time
print(f"Total Search Time: {elapsed_time_2:.2f} seconds")
print(strategy_accuracy)


strategy_type = "PostCompTraining"
sampler = RandomSampler()
print("PostCompTraining")
study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler,
)
start_time = time.time()
study.optimize(
    objective,
    n_trials=100,
    timeout=60 * 60 * 24,
)
end_time = time.time()
elapsed_time_3 = end_time - start_time
print(f"Total Search Time: {elapsed_time_3:.2f} seconds")
print(strategy_accuracy)



# {'RandomSampler': [0.8506, 0.8544, 0.85656, 0.86068, 0.85324, 0.84168, 0.85992, 0.86644, 0.83412, 0.84496, 0.8488, 0.8278, 0.83996, 0.86068, 0.85596, 0.83484, 0.85128, 0.8644, 0.83232, 0.8468, 0.84156, 0.82756, 0.85016, 0.82072, 0.84928, 0.74756, 0.82536, 0.83412, 0.82552, 0.85672, 0.82872, 0.85808, 0.81516, 0.84252, 0.8346, 0.861, 0.78672, 0.86252, 0.85784, 0.84256, 0.85636, 0.82972, 0.8602, 0.84536, 0.83212, 0.86152, 0.83152, 0.86492, 0.84752, 0.83896, 0.85964, 0.85284, 0.84848, 0.8462, 0.82412, 0.85876, 0.84224, 0.83128, 0.86036, 0.84052, 0.83972, 0.85112, 0.82576, 0.85672, 0.83612, 0.84448, 0.84732, 0.85364, 0.83948, 0.8388, 0.86508, 0.8408, 0.86424, 0.854, 0.835, 0.85544, 0.84616, 0.86036, 0.85664, 0.86416, 0.8604, 0.86088, 0.86028, 0.86812, 0.83816, 0.83992, 0.8474, 0.86616, 0.85952, 0.85996, 0.86112, 0.8448, 0.8566, 0.85656, 0.85596, 0.86108, 0.82852, 0.8554, 0.83304, 0.85804],
# 'TPESampler': [0.84616, 0.83508, 0.82768, 0.838, 0.83096, 0.83888, 0.85916, 0.85712, 0.84396, 0.85328, 0.86392, 0.86392, 0.86392, 0.86392, 0.85828, 0.85656, 0.86392, 0.86124, 0.86288, 0.86032, 0.85972, 0.86392, 0.85828, 0.8578, 0.86124, 0.86392, 0.86124, 0.86288, 0.85876, 0.84188, 0.86392, 0.86392, 0.86392, 0.86124, 0.85536, 0.85448, 0.85988, 0.85984, 0.86016, 0.86176, 0.84868, 0.86392, 0.86392, 0.86392, 0.86232, 0.83092, 0.84636, 0.86392, 0.85984, 0.80652, 0.85896, 0.86392, 0.86392, 0.86392, 0.86392, 0.85548, 0.85148, 0.84092, 0.86764, 0.85896, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.86508, 0.86072, 0.86764, 0.86764, 0.86764, 0.86764, 0.86764, 0.871, 0.85752, 0.85964, 0.84864, 0.86288, 0.871, 0.871, 0.871, 0.871, 0.871, 0.86288, 0.86916, 0.86916, 0.86824, 0.84664, 0.86824, 0.86824, 0.86824, 0.86824, 0.86488, 0.86404, 0.86824, 0.86488, 0.85792], 'GridSampler': []}


df = pd.DataFrame(strategy_accuracy)
df_cumulative = df.cummax()
df_cumulative["Trial"] = df.index + 1
csv_filename = "Compression_strategy_results.csv"
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
plt.savefig("Strategy_Comparison.png")

# Find the maximum accuracy reached by each sampler
best_accuracies = df_cumulative.drop(columns=["Trial"]).max()
# Find the best sampler (one with the highest final accuracy)
best_sampler = best_accuracies.idxmax()
best_accuracy_value = best_accuracies.max()
print(f"Best Sampler: {best_sampler} with accuracy {best_accuracy_value:.4f}")


with open("search_time.txt", "w") as file:
    file.write(str(elapsed_time_1) + "\n")
    file.write(str(elapsed_time_2) + "\n")
    file.write(str(elapsed_time_3) + "\n")

print("Elapsed time saved to 'search_time.txt'")