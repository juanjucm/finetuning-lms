# Fine-tuning LMs 🤖⚙️

This brief research is intended to explore the different fine-tuning approaches that can be applied for adapting bert-like models to a custom dataset and task.

<figure>
  <img src="./data/images/adaptive_fine-tuning.png">
  <figcaption style='text-align:center';>
  Framework for fine-tuning LMs. 
  <a href="https://ruder.io/recent-advances-lm-fine-tuning/">Sebastian Rude's post</a>
  </figcaption>
</figure>

### How to play

To run these notebooks, you will need **Python > 3.8.X**. Also, install the requirements.

```
pip install -r requirements.txt
```

## 1. Technical context
Let's make a very quick review of the different techniques we want to test:

1. **Base model training**: We will use a pretrained language model as an embedding generator. A simple FC layer will be trained for a classification task.

2. **Adaptive fine-tuning (augmented pre-training)**: Pretrained LM are robust in terms of o.o.d generalization. However, they still suffer when dealing with data belonging to an unseen domain. This training method involves fine-tuning the model so it can be adapted to a desired (new) domain.
    
    This is done using the same objectives as in the pre-training phase: **MLM** and **NSP** (BERT). This technique is useful to generate models when high performance is needed in a set of tasks within a specific data domain.

3. **Behavioural fine-tuning (task fine-tuning)**: This technique focus on addapting a model to a specific task. To do this, labeled data is required. In this set-up, the encoder's weights are unfreezed so the whole base model is trained along with the FC layer. Another option is to perform a previous step in which the model is trained with task-specific data related to the final objective.

For a wider explanation, please visit [Sebastian Ruder's amazing blog post](https://ruder.io/recent-advances-lm-fine-tuning/)

## 2. Hypothesis
Adaptive fine-tuning outperforms base model training. Behavioural fine-tuning outperforms adaptive fine-tuning. Adaptive + behavioural fine-tuning will achieve the best results.

## 3. Methodology
For conducting the experiments, proper dataset/task needs to be selected. The data needs to be different enough (in terms of domain) so the adaptation becomes a key aspect. Appart from that, we have to choose a base model and define the experiments, as well as define a set of metrics.

### 3.1. Dataset / Task
In order to properly test the influence of the fine-tuning method in the performance, we need to select a dataset (and a task) for which the adaptation plays a key rol. For example, selecting a very domain-specific dataset for solving sentiment analysis task might not be the best choice, since sentiment particles used to be not domain-related.

A good task for our case could be **Sentence Similarity classification**. In Sentence Similarity, the model needs to learn the semantics of the dataset so it can determine if both sentences are related or not. Working with this task allows us to use a dataset framed in a domain to which the model has not been exposed during the pre-training phase.

Taking that into consideration, the chosen dataset has been [medical_questions_pairs](https://huggingface.co/datasets/medical_questions_pairs). This dataset is composed by ~3K medical sentence pairs (questions). For each question, two different samples has been generated: 1) A reformulated sentence that entails with the original one. 2) A question not related with the original.

Being more specific on the numbers, the dataset contains a total of 3048 examples. We have splitted them into train and test. For training, we have assigned 2834 samples (93%). For testing, 214 (7%). For the purpose of this project, we don't need a third validation split.


### 3.2. Metrics
Since we are using Semantic Similarity in a classification setup as our target task, we have selected the usual classification metrics for evaluating the different trainings: accuracy, F1 score.

### 3.3. Model and training hyperparameters
Selected [bert-base-cased](https://huggingface.co/bert-base-cased) as our backbone model. For accessing and training the model, [HF transformers library](https://huggingface.co/docs/transformers/index) is being used.

For the experiments to be reproducible, I think it's worth to stablish the set of hyperparameters we will be using for training all experiments. You can see them in the table below.

<br>

| Hyperparameter       | Value |
| ----------- | -----------   |
| Per-device train batch size                       | 8 |
| Per-device eval batch size                        | 8 |
| Num train epochs                                  | 8 |
| Optimizer                                         | [AdamW](https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/optimizer_schedules#transformers.AdamW)
| Scheduler                                         | [Linear with warmup](https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup)
| Learning rate                                     | 2e-5 |
| Weight decay                                      | 0.01 |

<br>

Also, the hyperparameters used durign the **Adaptive fine-tuning** phase.

| Hyperparameter       | Value |
| ----------- | -----------   |
| Per-device train batch size                       | 16 |
| Per-device eval batch size                        | 16 |
| Num train epochs                                  | 8 |
| Optimizer                                         | [AdamW](https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/optimizer_schedules#transformers.AdamW)
| Scheduler                                         | [Linear with warmup](https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup)
| Learning rate                                     | 1e-5 |
| Weight decay                                      | 0.01 |

<br>

**NOTE: learning-rate is lower here since we don't want to 'erase' the encoder's pre-training knowledge. Just to get a subtle adaptation.**

These are the main hyperparameters. For the rest of them (dropout, num_hidden_layers, etc), we take the default value stablished in [TrainingArguments class](https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/trainer#transformers.TrainingArguments).

### 3.4. Experiment setup
Experiments will be conducted following this order.

1. First thing to do is to train our baseline. This is, freeze the encoder's weights and train just the FC layer on the classification task objective for a couple of epochs.

2. The second experiment is a test of **Behavioural fine-tuning**. This is the classic setup when fine-tuning a HF bert-like model. Just unfreeze backbone's weights and train them along with the FC layer for the task objective.

3. Next step, we performed **Adaptive fine-tuning**. In this phase, we trained the model with a ML objective (NSP will not be possible, since the dataset is composed by individual sentences). After that adaptation, a new version of the base encoder is generated. This time, taylored to our specifict semantic domain.

    <img src="./data/images/AFT.png" width="450">

    **During this step, new terms are not being added to the tokenizer's vocabulary. It would require the creation of random vectors in the encoder's embedding matrix. Probably this type of fine-tuning would allow the model to correctly train these new vectors. However, our dataset may not be big enough for that. We will let the model to learn the new vocabulary by understanding the relationships between the subtokens that the tokenizer is probably generating when encoding the unknown terms. Learn more about how BERT tokenizer works [here](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer)**.


4. Once we have our custom encoder, it's time to repeat the previous experiments and see if we achieve better results. First of all, again a basic training of the classification head. We will freeze our new encoder's weights and just train the upper dense layers.

    <img src="./data/images/A+HT.png" width="450">

5. Last experiment consists on a mix of both techniques. After an initial phase of Adaptative fine-tuning objective, we perform **Behavioural fine-tuning**. In this setup, our custom backbone's weights are trained twice, first for adapting to the data domain and then to further understand the specific task.

    <img src="./data/images/A+BFT.png" width="450">


**We have conducted the experiments in [Google Colab](https://colab.research.google.com/) environment**.

**For logging the experiments, storing the models and comparing results, we have used [Weight&Biases](https://wandb.ai/site)**.

## 4. Results

After training all different models, we can make a quick comparison. In the image below, you can see the accuracy metric evolution during the training process (8 epochs) for each experiment.

<figure>
  <img src="./data/images/wandb_accuracy_chart.png" width="550">
</figure>

**NOTE: We are just using accuracy, since F1-score results were not precise enough to describe the differences.**

<br>

If we extract top performances from each run, the final results are as follows:

<br>

| Model       | Best Accuracy | Best epoch   |
| ----------- | -----------   |  ----------- |
| Baseline (Head train)                                     | 68.22        | 8          |
| Behavioural fine-tuning                                   | 82.24        | 4          |
| Adaptative fine-tuning + Head train                       | 69.63        | 5          |
| **Adaptative fine-tuning + Behavioural fine-tuning**      | **85.05**    | **5**      |

<br>

First of all, our baseline training sets a reference score of 68.22% of accuracy.

Our next experiment, the behavioural fine-tuning over an 'stock' BERT checkpoint, achieves a 82.24% o accuracy. This shows how letting the backbone being adapted to the specific target task has great benefits on the result.

Last two experiments are performed after our adaptative fine-tuning run over the standard BERT checkpoint. During that process, we expected the model to be further specialiced in our specific domain, leading to better results on downstream tasks (like the classification problem we are tackling).

First, we re-trained our just-head experiment, this time using our custom backbone. Again, for this experiment we are freezing the backbone's weights and just trainig the classification head. Results are slightly better compared than the baseline test, getting a 69,63% of accuracy.

Last experiment, we just used our custom backbone to perform a behavioural fine-tuning. Following this strategy, in which we let the backbone to adjust its weights, we are training the whole thing to be adapted to our specific task. As expected, we got our best result with an accuracy score of 85.05%.

**Keep in mind that performance differences are relatively low. This is due to the small dataset we are using. With larger datasets, the difference between strategies would be probably higher**. 

Taking all of this into consideration, we have correctly confirmed our hypothesis. BERT-like models are a very powerful NLP modules that allows us to perform a wide variety of NLP tasks, even when they have not been explicitly trained from scratch to fit that task and/or semantic domain. A dual fine-tuning phase, first for adapting to the target domain and then for learning the objective task, would be a good way to extract all the potential out of these models.

## 5. References

* [Sebastian Ruder's post on fine-tuning strategies](https://ruder.io/recent-advances-lm-fine-tuning/)

* [Huggingface Transformers library](https://huggingface.co/docs/transformers/index)

* [Medical questions pairs dataset](https://huggingface.co/datasets/medical_questions_pairs)

* [Extra methods for adaptative fine-tuning](https://huggingface.co/course/chapter7/3?fw=tf#preprocessing-the-data)

* [How to increase BERT tokenizer's vocabulary](https://medium.com/@pierre_guillou/nlp-how-to-add-a-domain-specific-vocabulary-new-tokens-to-a-subword-tokenizer-already-trained-33ab15613a41)
