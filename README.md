# Source code for paper "Resolving Gendered Ambiguous Pronouns with BERT " 
[ArXiv](https://arxiv.org/abs/1906.01161), accepted at the 1st ACL Workshop on Gender Bias for Natural Language Processing at [ACL 2019](http://www.acl2019.org/EN/index.xhtml) in Florence on 2nd August 2019.

#### [Kaggle competition](https://www.kaggle.com/c/gendered-pronoun-resolution)
#### Abstract 
Pronoun resolution is part of coreference resolution, the task of pairing an expression to its referring entity. This is an important task for natural language understanding and a necessary component of machine translation systems, chat bots and assistants. Neural machine learning systems perform far from ideally in this task, reaching as low as 73\% F1 scores on modern benchmark datasets. Moreover, they tend to perform better for masculine pronouns than for feminine ones. Thus, the problem is both challenging and important for NLP researchers and practitioners. In this project, we describe our BERT-based approach to solving the problem of gender-balanced pronoun resolution. We are able to reach 92\% F1 score and a much lower gender bias on the benchmark dataset shared by Google AI Language team.

#### Navigation
 - `fine-tuned_model` containes the code reproducing Ken Krige's 5th place solution  
 - `frozen_model` containes the code reproducing 22th place solution by team "[ods.ai] five zeros" (Yury Kashnitsky, Matei Ionita, Vladimir Larin, Dennis Logvinenko, and Atanas Atanasov)
 - `analysis` folder containes scripts for comparison with results by Google AI Language team reported in [this paper](https://arxiv.org/abs/1810.05201)
