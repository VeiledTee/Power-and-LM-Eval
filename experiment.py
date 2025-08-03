import datasets
ds = datasets.load_dataset("hotpot_qa", "fullwiki", split="test")
print(ds[0])
