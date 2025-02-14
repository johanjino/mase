import warnings
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import chop.passes as passes
import warnings
from pathlib import Path
from chop import MaseGraph
from chop.tools import get_tokenized_dataset, get_trainer


class SuppressWarningsAndLogging:
    def __init__(self, logger_name=None, log_level=logging.ERROR):
        self.logger = logging.getLogger(logger_name) if logger_name else None
        self.log_level = log_level
        self.original_log_level = self.logger.level if self.logger else None

    def __enter__(self):
        # Suppress warnings
        self.warnings_context = warnings.catch_warnings()
        self.warnings_context.__enter__()
        warnings.simplefilter("ignore")

        # Suppress logging if a logger is specified
        if self.logger:
            self.logger.setLevel(self.log_level)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore warnings
        self.warnings_context.__exit__(exc_type, exc_value, traceback)

        # Restore original logging level
        if self.logger:
            self.logger.setLevel(self.original_log_level)


def gen_pruning_config(sparsity, strategy="l1-norm"):
    return {
                "weight": {
                    "sparsity": sparsity,
                    "method": strategy,
                    "scope": "local",
                },
                "activation": {
                    "sparsity": sparsity,
                    "method": strategy,
                    "scope": "local",
                },
            }


def sweepSparsity(step):

    checkpoint = "prajjwal1/bert-tiny"
    tokenizer_checkpoint = "bert-base-uncased"
    dataset_name = "imdb"

    mg = MaseGraph.from_checkpoint(f"{Path.home()}/adlsystems/tutorial_3_qat")

    dataset, tokenizer = get_tokenized_dataset(
        dataset=dataset_name,
        checkpoint=tokenizer_checkpoint,
        return_tokenizer=True,
    )

    trainer = get_trainer(
        model=mg.model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
    )

    eval_results = trainer.evaluate()
    baseline_accuracy = eval_results['eval_accuracy']

    random_accuracy = []
    random_retrain_accuracy = []

    l1norm_accuracy = []
    l1norm_retrain_accuracy = []

    sparsity_array = [float(sparsity)/10 for sparsity in range(1, 10, step)]
    
    for strategy in ["l1-norm", "random"]:
        for sparsity in range(1, 10, step):
            fpSparsity = float(sparsity)/10
            print("Sparsity:", fpSparsity)

            mg = MaseGraph.from_checkpoint(f"{Path.home()}/adlsystems/tutorial_3_qat")
            
            pruning_config = gen_pruning_config(fpSparsity, strategy)
            mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)
            
            trainer = get_trainer(
                model=mg.model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=5,
            )

            eval_results = trainer.evaluate()
            if strategy == "random":
                random_accuracy.append(eval_results['eval_accuracy'])
            elif strategy == "l1-norm":
                l1norm_accuracy.append(eval_results['eval_accuracy'])
            else:
                raise TypeError("Strategy not implemented")

            trainer.train()

            eval_results = trainer.evaluate()
            if strategy == "random":
                random_retrain_accuracy.append(eval_results['eval_accuracy'])
            elif strategy == "l1-norm":
                l1norm_retrain_accuracy.append(eval_results['eval_accuracy'])
            else:
                raise TypeError("Strategy not implemented")
        

            


    print(random_accuracy)
    print(random_retrain_accuracy)
    print(l1norm_accuracy)
    print(l1norm_retrain_accuracy)
    print(sparsity)

    # Create a DataFrame with separate columns for PTQ and QAT accuracy
    df = pd.DataFrame({
        "sparsity": sparsity_array,
        "random pruning": random_accuracy,
        "retrained random pruning": random_retrain_accuracy,
        "l1Norm pruning": l1norm_accuracy,
        "retrained l1Norm pruning": l1norm_retrain_accuracy
    })

    # Save DataFrame to CSV
    csv_filename = "pruning_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

    # Define styles manually
    styles = {
        "random pruning": {"color": "red", "marker": "o", "alpha": 1, "linestyle": "dashed"},
        "retrained random pruning": {"color": "blue", "marker": "x", "alpha": 1, "linestyle": "dashed"},
        
        "l1Norm pruning": {"color": "green", "marker": "o", "alpha": 1, "linestyle": "dashed"},
        "retrained l1Norm pruning": {"color": "orange", "marker": "x", "alpha": 1, "linestyle": "dashed"},
    }

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.axhline(y=baseline_accuracy, color="cornflowerblue", linestyle="dashed", linewidth=2, label="Baseline", alpha=0.75)

    # Manually scatter plot each method and connect points with dashed lines
    for method, style in styles.items():
        plt.scatter(df["sparsity"], df[method], 
                    color=style["color"], marker=style["marker"], alpha=style["alpha"],
                    s=50, label=method)
        
        # Connect points with a dashed line
        # plt.step(df["sparsity"], df[method], 
        #          color=style["color"], linestyle=style["linestyle"], alpha=0.5)

    # Customize plot
    plt.xlabel("Sparsity Ratio", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Pruning Methods Accuracy Comparison", fontsize=14)
    plt.grid(True)
    plt.legend(title="Method")
    plt.savefig("Pruning.png")





# Main
with SuppressWarningsAndLogging("accelerate.utils.other", logging.ERROR):
    sweepSparsity(1)
