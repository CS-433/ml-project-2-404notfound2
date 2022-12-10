# Commensense Causal Reasoning

Basic Intro

https://lia.epfl.ch/wp-content/uploads/project_proposals/proposal_498.pdf

Steps:

1. Run BERT on COPA, finetuning
2. Propose new prompts
3. Generation tasks with/without prompts
......

## Finetuning on Classification Task

```shell
sh copa_classification.sh
```

We compare `bert-base-uncased`, `roberta-base`, and `xlm-roberta-base`.

Result (bad):

|           | BERT-base-uncased | RoBERTa-base | XLMRoBERTa-base |
| --------- | ----------------- | ------------ | --------------- |
| Accuracy  |          $0.00\pm0.00$         |       $0.00\pm0.00$       |        $0.00\pm0.00$         |
| Precision |           $0.00\pm0.00$        |        $0.00\pm0.00$      |         $0.00\pm0.00$        |
| Recall    |            $0.00\pm0.00$       |         $0.00\pm0.00$     |         $0.00\pm0.00$        |
| F1-score  |          $0.00\pm0.00$         |        $0.00\pm0.00$      |          $0.00\pm0.00$       |

Existing code:

- [COPA Benchmark (Question Answering) | Papers With Code](https://paperswithcode.com/sota/question-answering-on-copa?p=deberta-decoding-enhanced-bert-with)
- [Andrew S. Gordon: Choice of Plausible Alternatives (usc.edu)](https://people.ict.usc.edu/~gordon/copa.html)
- [Replicating Roberta's results on SuperGlue COPA task · Issue #3051 · facebookresearch/fairseq (github.com)](https://github.com/facebookresearch/fairseq/issues/3051)
