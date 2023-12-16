from transformers import pipeline

pipe = pipeline(model="facebook/bart-large-mnli")
print(pipe("Even if by a special privilege their predestination were revealed to some, it is not fitting that it should be revealed to everyone; because, if so, those who were not predestined would despair; and security would beget negligence in the predestined.",
     candidate_labels=["insane", "sane"]))


