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

We compare `bert-base-uncased`, `roberta-base`, `xlm-roberta-base`, `albert-base-v2`, and `albert-large-v2`.

Result (bad):

|           | BERT-base-uncased | RoBERTa-base | XLM-RoBERTa-base | ALBERT-base-v2 | ALBERT-large-v2 |
| --------- | ----------------- | ------------ | --------------- | --------------- | --------------- |
| Accuracy  |          $\bm{74.60}\pm1.20$         |       $68.40\pm1.20$       |        $64.20\pm1.60$         |        $69.60\pm0.80$         |        $\bm{74.60}\pm1.20$         |
| Precision |           ${74.34}\pm1.21$        |        $68.35\pm1.30$      |         $66.05\pm2.29$        |        $69.62\pm0.97$         |        $\bm{74.35}\pm1.22$         |
| Recall    |            $\bm{74.36}\pm1.25$       |         $68.53\pm1.33$     |         $65.39\pm1.39$        |        $69.78\pm1.05$         |        $74.32\pm1.17$         |
| F1-score  |          $\bm{74.35}\pm1.23$         |        $68.29\pm1.57$      |          $64.06\pm1.53$       |        $69.50\pm0.91$         |        $74.33\pm1.19$         |

## Finetuning on Classification Task

```shell
sh copa_classification.sh
```

We compare `bert-base-uncased`, `roberta-base`, `xlm-roberta-base`, and `bart-base`.

Result (bad):

|           | BERT-base-uncased | RoBERTa-base | XLM-RoBERTa-base | BART-base |
| --------- | ----------------- | ------------ | --------------- | --------------- |
| Perplexity  |          ${11.8303}$         |       $8.0866$       |        $10.5570$         |        $\bm{1.0357}$         |
| BLEU-1 |           ${0.0200}$        |        $\bm{0.2671}$      |         $0$        |        $0.0199$         |
| BLEU-2    |            ${0.0054}$       |         $\bm{0.0594}$     |         $0$        |        $0.0057$         |
| BLEU-3  |          $\bm{0}$         |        $0$      |          $0$       |        $\bm{0.0020}$         |
| BLEU-4  |          $\bm{0}$         |        $0$      |          $0$       |        $0$         |
| METEOR  |          ${0.0890}$         |        $\bm{0.1648}$      |          $0.0079$       |        $0.0873$         |
| ROUGE-L  |          ${0.0428}$         |        $\bm{0.1497}$      |          $0.0080$       |        $0.0064$         |
| CIDEr  |          ${0.0008}$         |        $\bm{0.1090}$      |          $0.003$       |        $0.0091$         |

Existing code:

- [COPA Benchmark (Question Answering) | Papers With Code](https://paperswithcode.com/sota/question-answering-on-copa?p=deberta-decoding-enhanced-bert-with)
- [Andrew S. Gordon: Choice of Plausible Alternatives (usc.edu)](https://people.ict.usc.edu/~gordon/copa.html)
- [Replicating Roberta's results on SuperGlue COPA task · Issue #3051 · facebookresearch/fairseq (github.com)](https://github.com/facebookresearch/fairseq/issues/3051)
