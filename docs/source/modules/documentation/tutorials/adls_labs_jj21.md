# ADLSys Coursework

## Lab 0: Introduction to Mase

### Tutorial 1

### Tutorial 2

## Lab 1: Model Compression (Quantization and Pruning)

### Tutorial 3: QAT on Bert

**Tasks:** 

Explore a range of fixed point widths from 4 to 32. Plot a figure where the x-axis is the fixed point width and the y-axis is the highest achieved accuracy on the IMDb dataset

Plot separate curves for PTQ and QAT at each precision to show the effect of post-quantization finetuning.




### Tutorial 4: Unstructed pruning on Bert

**Task:** Vary the sparsity from 0.1 to 0.9.

Plot a figure where the x-axis is the sparsity and the y-axis is the highest achieved accuracy on the IMDb dataset

Plot separate curves for Random and L1-Norm methods to evaluate the effect of different pruning strategies.



## Lab 2: Neural Architecture Search

### Tutorial 5: NAS with Mase and Optuna

**Tasks:** 

Explore using the GridSampler and TPESampler in Optuna. Plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each sampler to compare their performance.

Perform compression aware search flow. Plot a new figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. There should be three curves: 1. without compression, compression-aware search, and compression-aware search with post-compression training.


## Lab 3: Mixed Precision Search

### Tutorial 6: Mixed Precision Quantization Search with Mase and Optuna

**Tasks:** 

Explore different layers to have widths in the range [8, 16, 32] and fractional widths in the range [2, 4, 8]. Plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis.

Extend the search to consider all supported precisions for the Linear layer in Mase, including Minifloat, BlockFP, BlockLog, Binary, etc. Plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each precision to compare their performance.


