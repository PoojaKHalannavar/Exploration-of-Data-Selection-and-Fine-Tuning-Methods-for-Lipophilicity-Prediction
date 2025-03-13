# Task 3: Exploration of Data Selection and Fine-Tuning Methods

In this task, we explore alternative methods for data selection and investigate several fine-tuning techniques to adapt a pre-trained model for a specific task. The goal is to improve the model performance on our target dataset.

Data Selection Strategies: The first step in fine-tuning a model is to carefully select the training data. While the previous tasks focused on influence-based data selection, here we experiment with Cosine Similarity. 

Pick one data selection method by yourself. Log your findings about the selected data subsets:
How much data is used in each strategy?
Compare the performance of models trained with each selection method.

Fine-tuning Strategies :In this section, we implement and compare 3 parameter-efficient fine-tuning approaches:
1. bitfit
2. LoRA (Low-Rank Adaptation)
3. iA3 (Implicit Adapter)
