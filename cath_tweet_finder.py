from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
sequence_to_classify = "The only innocent feature in babies is the weakness of their frames; the minds of infants are far from innocent."
candidate_labels = ['absurd', 'rational']
print(classifier(sequence_to_classify, candidate_labels))


