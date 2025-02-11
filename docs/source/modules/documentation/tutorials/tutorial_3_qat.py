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


def generate_quantization_config(bit_width):
    """
    Generates a quantization configuration for a given bit-width.
    """
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
                # Data configuration
                "data_in_width": bit_width,
                "data_in_frac_width": bit_width // 2,  # Half for fractional part
                # Weight configuration
                "weight_width": bit_width,
                "weight_frac_width": bit_width // 2,  # Half for fractional part
                # Bias configuration
                "bias_width": bit_width,
                "bias_frac_width": bit_width // 2,  # Half for fractional part
            }
        },
    }


def sweepBitWidth(start, end, step):

    checkpoint = "prajjwal1/bert-tiny"
    tokenizer_checkpoint = "bert-base-uncased"
    dataset_name = "imdb"

    mg = MaseGraph.from_checkpoint(f"{Path.home()}/adlsystems/tutorial_2_lora")

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

    PTQ_accuracy = []
    QAT_accuracy = []
    widths = []
    for width in range(start, end+1, step):
        print("Running: ", width)
        widths.append(width)

        mg = MaseGraph.from_checkpoint(f"{Path.home()}/adlsystems/tutorial_2_lora")

        quantization_config = generate_quantization_config(width)

        mg, _ = passes.quantize_transform_pass(
            mg,
            pass_args=quantization_config,
        )

        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
        )

        eval_results = trainer.evaluate()
        PTQ_accuracy.append(eval_results['eval_accuracy'])

        trainer.train()

        eval_results = trainer.evaluate()
        QAT_accuracy.append(eval_results['eval_accuracy'])


    print(PTQ_accuracy)
    print(QAT_accuracy)
    print(widths)
    # [0.5, 0.76548, 0.86656, 0.86808, 0.865, 0.86864, 0.8682, 0.86868, 0.86912, 0.86912, 0.86868, 0.8688, 0.86876, 0.86872, 0.86876]
    # [0.5, 0.87192, 0.87492, 0.87508, 0.87452, 0.8758, 0.87668, 0.8762, 0.87564, 0.8762, 0.8762, 0.87596, 0.87568, 0.87536, 0.87568]
    # [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

    # Create a DataFrame with separate columns for PTQ and QAT accuracy
    df = pd.DataFrame({
        "Width": widths,
        "PTQ_Accuracy": PTQ_accuracy,
        "QAT_Accuracy": QAT_accuracy
    })

    # Save DataFrame to CSV
    csv_filename = "PATandQAT_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

    # Convert DataFrame to long format for Seaborn plotting
    df_melted = df.melt(id_vars=["Width"], var_name="Method", value_name="Accuracy")

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.axhline(y=baseline_accuracy, color="red", linestyle="dashed", linewidth=2, label="Baseline", alpha=0.45)
    sns.scatterplot(data=df_melted, x="Width", y="Accuracy", hue="Method", style="Method", s=100)

    # Customize plot
    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("PTQ vs QAT Accuracy", fontsize=14)
    plt.grid(True)
    plt.legend(title="Method")
    plt.savefig("PTQandQAT.png")





# Main
with SuppressWarningsAndLogging("accelerate.utils.other", logging.ERROR):
    min_bit_width = 4
    max_bit_width = 32
    sweepBitWidth(min_bit_width, max_bit_width, 2)
