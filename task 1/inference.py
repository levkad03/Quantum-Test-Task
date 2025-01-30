from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

model_path = "task 1/saved_model"

# id to label for better understanding
id2label = {0: "O", 1: "B-MNTN", 2: "I-MNTN"}

# define model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(model_path, id2label=id2label)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# define pipeline for model inference
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

sentence = "The Aconcagua is the highest mountain in the world."

results = ner_pipeline(sentence)

print(results)
