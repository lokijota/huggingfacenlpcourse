# Random readings pursued from the main course

## Chain of Thought Reasoning

- Source: Google - Chain of Thought Reasoning. Overview here https://ai.googleblog.com/2022/05/language-models-perform-reasoning-via.html and ArXiv here: https://arxiv.org/abs/2201.11903

(notes added to PDF file)

Key ideas/abstract:
- We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain-of-thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain-of-thought prompting improves performance on a range of **arithmetic, commonsense, and symbolic reasoning tasks**. The empirical gains can be striking. For instance, prompting a PaLM 540B with just eight chain-of-thought exemplars achieves state-of-the-art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.
- chain of thought—a coherent series of *intermediate reasoning steps* that lead to the final answer for a problem
- *Chain of thought reasoning allows models to decompose complex problems into intermediate steps that are solved individually. Moreover, the language-based nature of chain of thought makes it applicable to any task that a person could solve via language. We find through empirical experiments that chain of thought prompting can improve performance on various reasoning tasks, and that successful chain of thought reasoning is an emergent property of model scale — that is, the benefits of chain of thought prompting only materialize with a sufficient number of model parameters (around 100B).*
- Employing chain of thought prompting enables language models to solve **arithmetic reasoning problems** for which standard prompting has a mostly flat scaling curve.
- chain-of-thought prompting does not positively impact performance for small models, and only yields performance gains when used with **models of
∼100B parameters**. We qualitatively found that *models of smaller scale produced fluent but illogical chains of thought*, leading to lower performance than standard prompting.

Results for mathematical problems:
- models of smaller scale produced fluent but illogical chains of thought, leading to lower performance than standard prompting
- chain-of-thought prompting has larger performance gains for more-complicated problems.
- chain-of-thought prompting via GPT-3 175B and PaLM 540B compares favorably to prior state of the art, which typically finetunes a task-specific model on a labeled training dataset.

Results for Common sense reasoning
- For all tasks, scaling up model size improved the performance of standard prompting; chain-of-thought prompting led to further gains, with improvements appearing to be largest for PaLM 540B [...] These results demonstrate that chain-of-thought prompting can also improve performance on tasks
requiring a range of commonsense reasoning abilities

Notes from discussion:

- For many reasoning tasks where standard prompting has a flat scaling curve, chain-of-thought prompting leads to dramatically increasing scaling curves. *Chain-of-thought prompting appears to expand the set of tasks that large language models can perform successfully* — in other words, our work underscores that *standard prompting only provides a lower bound on the capabilities of large language models*. This observation likely raises more questions than it answers — for instance, *how much more can we expect reasoning ability to improve with a further increase in model scale? What other prompting methods might expand the range of tasks that language models can solve?*

And notes from FAQ:
- Examples of semantic understanding and one-step missing errors that were fixed by scaling PaLM to 540B are given in Figure 10. *This result appears consistent with a hypothesis that language models acquire a range of semantic understanding and logical reasoning skills as a function of model scale*.