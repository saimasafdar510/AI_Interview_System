# Experimental answer-classifier data

`answer_classifier_training.csv` is a hand-authored, illustrative dataset created for exercising the experimental text-classification pipeline. It is not collected from candidates, independently annotated, validated, or representative of real interview answers.

Each row contains a `question_group`, an answer text, and a `label`:

- `poor`: very brief or vague text with little explanation.
- `average`: a relevant but partial explanation.
- `good`: a more developed explanation, often including an example, trade-off, or caveat.

These labels describe the examples' apparent completeness for this demonstration. They do not establish factual correctness, interview performance, or candidate ability. The labels are subjective and intentionally simplified.

The training command holds out entire `question_group` values, so answers to the same prompt cannot appear in both the training and test partitions. The dataset is small and synthetic; held-out metrics are only a pipeline sanity check and must not be interpreted as validated generalization.
