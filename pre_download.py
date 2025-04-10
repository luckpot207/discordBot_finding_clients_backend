from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",
    token="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"
)

labels = ["hiring freelancer", "spam", "chat"]
result = classifier("Looking for an AI engineer for a short project.", candidate_labels=labels)

print(result["labels"][0], result["scores"][0])
