from lm_eval.api.registry import get_model
from lm_eval.evaluator import evaluate  # or "from lm_eval import simple_evaluate"

# Step 1: instantiate the “local-completions” class using the same arg string as CLI:
lm = get_model("local-completions") \
    .create_from_arg_string(
        "model=gemma3:1b,"
        "base_url=http://localhost:11434/v1/completions,"
        "num_concurrent=8,"
        "tokenized_requests=true"
    )

# Step 2: programmatically evaluate tasks (e.g. truthfulness, commonsense)
results = evaluate(
    lm=lm,
    tasks=["truthfulqa", "hellaswag"],
    num_fewshot=0,
    batch_size="auto",
    apply_chat_template=True,
    output_path="results.json",
    log_samples=True
)

print(results)
