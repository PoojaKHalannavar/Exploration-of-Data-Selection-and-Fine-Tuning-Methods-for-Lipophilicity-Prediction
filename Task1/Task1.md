# Task 1: Supervised and Unsupervised Fine-tuning of a Chemical Language Model on Lipophilicity

a. Supervised Finetuning
The goal is to fine-tune a pre-trained chemical language model on a regression task using the Lipophilicity dataset. The task involves predicting the lipophilicity value for a given molecule representation (SMILES string).
b. Add Unsupervised Finetuning
In this step, we performed unsupervised fine-tuning on the training dataset. This means the model will leverage only the SMILES strings without any corresponding labels to adapt its understanding of the data distribution. By familiarizing the model with the patterns and structure of the SMILES strings, you can potentially enhance its performance on downstream supervised tasks.
For this fine-tuning, we used the Masked Language Modeling (MLM) objective, where the model learns to predict randomly masked tokens within the input sequence. Here we saved the fine-tuned model for later use in Task2 and Task3.
c. Fine-Tune for Comparison
After performing unsupervised fine-tuning on the training data, we now fine-tune the model on the regression task with the regression head. By comparing the performance of the model before and after unsupervised fine-tuning, we evaluate how the unsupervised fine-tuning impacts the model's performance on our target task.
