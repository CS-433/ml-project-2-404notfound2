# Fine-tuning and Prompt-learning on Commonsense Causal Reasoning (CCR)

The second [project](https://lia.epfl.ch/wp-content/uploads/project_proposals/proposal_498.pdf) of EPFL Machine learning course (CS-433). We choose the ML4Science option for the machine learning project.

## Team Members

- Yiyang Feng: yiyang.feng@epfl.ch
- Naisong Zhou: naisong.zhou@epfl.ch
- Yuheng Lu: yuheng.lu@epfl.ch

## Abstract

Commonsense Causal Reasoning (CCR) aims to understand and reason about the cause-and-effect relationships in the world. The [COPA](https://people.ict.usc.edu/~gordon/copa.html) dataset is widely used to evaluate the performance of systems in CCR tasks. In this project, we define the COPA CCR task into two sub-tasks: the original classification task and the cause/effect generation task. We then implement fine-tuning models and the prompt learning model GPT-3 on these sub-tasks. Finally, we compare the performance between these models, and the results have shown that these models learn some commonsense causal relationship, and the performance of GPT-3 with prompt learning is significantly better on both tasks. We analyze the superior performance of GPT-3 may be due to more of its large number of model parameters and massive pre-trained dataset than prompt learning itself. However, the ability of few-shot learning is still important for its efficiency in down-stream adaptation.

## Install requirements

```shell
conda create -n mlproject2 python=3.8 jupyterlab=3.2 numpy=1.23.5 transformers=4.20.1 tqdm=4.64.1 evaluate=0.3.0 pandas=1.5.2 wandb=0.12.21 datasets=2.7.1
conda activate mlproject2
```

For the PyTorch environment, just reference the [website](https://pytorch.org/) and choose the suitable version according your cuda version.

## Dataset

COPA, short for Choice Of Plausible Alternatives, is a widely used dataset in evaluating performance in CCR tasks. Here is an example for illustration:

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/cd5cf79e2d45e04b14f385f9633d6ea0e1658db414c96c12f4a6f4eaa546a117.png" />

It has five fields. One premise, two choices, one question and one label. The original COPA task is just a binary classification task.

We don't have to download the dataset manually, just import from the class `datasets`.

```python
from datasets import load_dataset
copa = load_dataset("super_glue", "copa")
```

## Task Definition

We divide the CCR task into two sub-tasks: text classification and generation. Here is the illustration of the two subtasks.

|   <img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/65c2d1f066d02a04b7830d29039a505d9173d1f6aa1b268fc4e080ac1fbb3d8b.png" /><br>classification   |   <img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/358d06c18440e5550eac94f976598881ed1c9fbe313bb630ccadc85770f18c23.png" /><br>generation   |
| ---- | ---- |

## Experiment Settings

We study the fine-tuning and prompt-learning method for these tasks, the two main methods for current language models.

## Classification Task

We compare `bert-base-uncased`, `roberta-base`, `xlm-roberta-base`, `albert-base-v2`, `albert-large-v2`, and `GPT-3-text-davinci-003`.

For fine-tuning models, run:

```shell
sh copa_classification.sh
```

For prompt-learning models, run the notebook `copa_classification_prompt_learning.ipynb`. Add a key `api_key.txt` in the main folder for storing the [OpenAI API key](https://beta.openai.com/account/api-keys).

Result:

- *Italics:* best fine-tuning performance
- **Bold:** best performance

|           | BERT<br>-base<br>-uncased | RoBERTa<br>-base | XLM<br>-RoBERTa<br>-base | ALBERT<br>-base<br>-v2 | ALBERT<br>-large<br>-v2 | GPT-3<br>-175B |
| --------- | ----------------- | ------------ | --------------- | --------------- | --------------- | --------------- |
| Accuracy  |      *74.6±1.2*      |       68.4±1.2       |        64.2±1.6        |        69.6±0.8        |        *74.6±1.2*        |        **92.0±0.6**        |
| Precision | 74.3±1.2          |      68.4±1.3      |        66.1±2.3        |        69.6±1.0        |        *74.4±1.2*        |        **92.4±0.7**        |
| Recall    |       *74.4±1.3*       |     68.5±1.3     |        65.4±1.1        |        69.8±1.0        |        74.3±1.2        |        **91.6±0.6**        |
| F1-score  |         *74.4±1.2*         |      68.3±1.6      |       64.1±1.5       |        69.5±0.9        |        74.3±1.2        |        **91.8±0.6**        |

## Generation Task

We compare `bert-base-uncased`, `roberta-base`, `xlm-roberta-base`, and `bart-base`.

For fine-tuning models, run:

```shell
sh copa_generation.sh
```

For prompt-learning models, run the notebook `copa_generation_prompt_learning.ipynb`. Add a key `api_key.txt` in the main folder as above.

Result:

- *Italics:* best fine-tuning performance
- **Bold:** best performance

|           | BERT<br>-base<br>-uncased | RoBERTa<br>-base | XLM<br>-RoBERTa<br>-base | BART<br>-base | GPT-3<br>-175B |
| --------- | ----------------- | ------------ | --------------- | --------------- | --------------- |
| BLEU-1 |        2.0%        |      26.7%      |         0        |   *31.2%*   |   **39.2%**   |
| BLEU-2    |         0.5%       |     5.9%     |         0        |        *11.8%*        |        **23.3%**        |
| BLEU-3  |         0         |        0      |          0       |   *6.2%*   | **14.3%** |
| BLEU-4  |         0         |        0      |          0       |   *3.5%*   |   **14.8%**   |
| METEOR  |         8.9%         |      16.5%      |       0.8%       |   *23.3%*   |   **24.0%**   |
| ROUGE-L  |         4.3%         |      15.0%      |        0.8%       |   **22.8%**   |   15.1%   |
| CIDEr  |         0.001         |      0.109      |          0.003       |   *0.399*   |   0.892   |

## Files

- `copa_classification.py` : CCR classification script for the fine-tuning models.
- `copa_generation.py` : CCR generation script for the fine-tuning models.
- `copa_classification.sh` : configuration file for the fine-tuning models on CCR classification. comments in the shell script show how to use the script.
- `copa_generation.sh` : configuration file for the fine-tuning models on CCR generation. comments in the shell script show how to use the script.
- `report.pdf` : our report for this project.
- `copa_classification_prompt_learning.ipynb` : CCR classification notebook for the prompt-learning models.
- `copa_generation_prompt_learning.ipynb` : CCR generation notebook for the prompt-learning models.
- `api_key.txt`: your secret OpenAI API key for running the GPT-3 model.
- `cider` : folder for CIDEr score. (no supported in the official libraries.)
  - `cider_scorer.py` : cider score computation.
  - `cider.py` : define a class for computing cider score.
