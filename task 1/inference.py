from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

model_path = "saved_model"

id2label = {0: "O", 1: "B-MNTN", 2: "I-MNTN"}

model = AutoModelForTokenClassification.from_pretrained(model_path, id2label=id2label)
tokenizer = AutoTokenizer.from_pretrained(model_path)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

sentence = "The Mount Everest is the highest mountain in the world."

results = ner_pipeline(sentence)

print(results)
