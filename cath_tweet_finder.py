from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
sequence_to_classify = "Gravity exists because all things desire unity, and the greater another thing is, the more that things desire unity with it."
candidate_labels = ['Theistic', 'Atheistic']
print(classifier(sequence_to_classify, candidate_labels))


