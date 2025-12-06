from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main")
print(len(dataset["train"]))
