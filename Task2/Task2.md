# Task 2: Influence Function-based Data Selection


Now we understood how our model performs on Lipophilicity dataset through `Task1`. The goal in this task is to further enhance the performance by selecting external datapoints for training.

Here is an external dataset (../tasks/External-Dataset_for_Task2.csv) with molecular SMILES strings and corresponding lipophilicity values that we include in the training process. However, we suspect that not all external data points are relevant. So we aim to only select those that will likely improve the model's performance.  To achieve this, we used **influence functions** to compute the impact of each external data point on the model’s behavior. This will help us identify the most valuable data points for training. By influence functions, we can analyze the distribution of influence scores and identify the high-impact samples. For influence computation, we refer to [Koh & Liang’s paper (2017)](https://arxiv.org/abs/1703.04730) on influence functions to calculate and log the influence scores for all samples in the external dataset. 

The calculation of the influence function involves three main steps: computing the gradient of the training loss with respect to the model parameters, estimating the inverse of the Hessian matrix, and combining these to evaluate the effect of the training point on the test loss. The challenge for using it in deep neural networks is that storing and inverting the Hessian requires \( O(d^3) \) operations, where \( d \) is the number of model parameters, making it infeasible to compute for large neural networks. To address this, Koh & Liang (2017) proposed approximating the inverse Hessian-vector product (iHVP) using techniques like **Stochastic estimation/LiSSA** [(Agarwal et al., 2016)](https://arxiv.org/abs/1602.03943).

1. Our task here was to compute the influence scores for each data point in the external dataset using the LiSSA approximation. This will help us identify which external samples are most influential in improving the model's performance. For this, we:
- use the trained model and the external dataset.
- compute the gradients for each data point in the external dataset.
- use the LiSSA approximation to estimate the influence of each external sample on the model's performance on the test set.

2. Once we got the influence scores for the external dataset, we combine the high-impact samples selected with the Lipophilicity training dataset and fine-tune the model again. We then evaluate the model’s performance on the Lipophilicity test set and compare it to the baseline in `Task1`.
