import datasets
ds = datasets.load_dataset("hotpot_qa", "fullwiki", split="validation")
print(ds[0])

# CLI command to run full hotpot validation set
### Linux CLI
# lm_eval --model local-chat-completions --tasks hotpotqa_fullwiki --log_samples --output_path ./results/gemma3:1b/ --write_out --trust_remote_code --model_args 'model=gemma3:1b,base_url=http://localhost:11434/v1/chat/completions/,num_concurrent=1,max_retries=3,tokenized_requests=False,max_length=8192,apply_chat_template=True,eos_string='"'"'<end_of_turn>'"'"',tokenizer=google/gemma-2b-it' --batch_size 4 --apply_chat_template --include_path lm_eval/tasks/hotpotqa_fullwiki --trust_remote_code --apply_chat_template
### Windows CLI
# lm_eval --model local-chat-completions --tasks triviaqa --log_samples --output_path ./results/gemma3:1b/ --write_out --trust_remote_code --model_args "model=gemma3:1b,base_url=http://localhost:11434/v1/chat/completions/,num_concurrent=1,max_retries=3,tokenized_requests=False,max_length=8192,apply_chat_template=True,eos_string='<end_of_turn>',tokenizer=google/gemma-2b-it" --batch_size 1 --apply_chat_template --trust_remote_code --apply_chat_template --limit 5
