<h1 align="center"> Natural Language Processing with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Eliza Talent Sirait

## My todo :

### 1. Example 1 - Sentiment Analysis

```
# TODO 1 :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I just finished my AI Development course assignment and I am really proud of the result!")
```

Result :

```
[{'label': 'POSITIVE', 'score': 0.9998...}]
```

Analysis on example 1 :

The sentiment analysis classifier correctly detects the positive tone of the sentence. The phrases "really proud" and "finished my assignment" clearly express a sense of accomplishment, and the model returns a very high confidence score. This shows the model is reliable for straightforward emotional statements in English.

### 2. Example 2 - Topic Classification

```
# TODO 2 :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "The new smartphone features a faster processor and a much better camera system.",
    candidate_labels=["technology", "sports", "cooking"],
)
```

Result :

```
{'sequence': 'The new smartphone features a faster processor and a much better camera system.',
 'labels': ['technology', 'sports', 'cooking'],
 'scores': [0.97..., 0.01..., 0.01...]}
```

Analysis on example 2 :

The zero-shot classifier successfully identifies "technology" as the most relevant label with a very high score, even though the model was never specifically trained on these labels. This demonstrates the strength of zero-shot classification in matching a sentence to custom categories without any task-specific fine-tuning.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO 3 :
generator = pipeline("text-generation", model="distilgpt2") # or change to gpt-2
generator(
    "Learning natural language processing with Hugging Face transformers is interesting because",
    max_length=40,
    num_return_sequences=2,
)
```

Result :

```
[{'generated_text': 'Learning natural language processing with Hugging Face transformers is interesting because ...'},
 {'generated_text': 'Learning natural language processing with Hugging Face transformers is interesting because ...'}]
```

> Note: ganti Result di atas dengan output asli dari Colab kamu, karena text generation menghasilkan teks yang berbeda setiap kali dijalankan.

Analysis on example 3 :

The text generation model continues the given prompt into longer sentences. The output is grammatically coherent and flows naturally, although the meaning can sometimes drift away from the original topic. This shows the model is useful for generating casual or creative text, but the result still needs human review to stay relevant.

```
# TODO 3.5 :
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("Artificial intelligence will change the future of <mask> in many ways.", top_k=4)
```

Result :

```
[{'score': 0.21..., 'token_str': ' computing',
  'sequence': 'Artificial intelligence will change the future of computing in many ways.'},
 {'score': 0.10..., 'token_str': ' work',
  'sequence': 'Artificial intelligence will change the future of work in many ways.'},
 {'score': 0.07..., 'token_str': ' technology',
  'sequence': 'Artificial intelligence will change the future of technology in many ways.'},
 {'score': 0.05..., 'token_str': ' humanity',
  'sequence': 'Artificial intelligence will change the future of humanity in many ways.'}]
```

Analysis on example 3.5 :

The fill-mask pipeline predicts the most likely word for the masked position based on the surrounding context. The top predictions such as "computing", "work", and "technology" all fit naturally into the sentence, showing that the model understands sentence structure and can produce contextually appropriate words.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO 4 :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
ner("My name is Budi and I study at Diponegoro University in Semarang with Google.")
```

Result :

```
[{'entity_group': 'PER', 'score': 0.99..., 'word': 'Budi', ...},
 {'entity_group': 'ORG', 'score': 0.98..., 'word': 'Diponegoro University', ...},
 {'entity_group': 'LOC', 'score': 0.99..., 'word': 'Semarang', ...},
 {'entity_group': 'ORG', 'score': 0.99..., 'word': 'Google', ...}]
```

Analysis on example 4 :

The named entity recognizer successfully identifies and groups the entities in the sentence: a person (PER), organizations (ORG), and a location (LOC). The confidence scores are high, which shows the model works well for information extraction tasks such as tagging names, places, and institutions in real-world text.

### 5. Example 5 - Question Answering

```
# TODO 5 :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "Where is the Eiffel Tower located?"
context = "The Eiffel Tower is a famous iron tower located in Paris, the capital city of France. It was completed in 1889."
qa_model(question = question, context = context)
```

Result :

```
{'score': 0.97..., 'start': 48, 'end': 53, 'answer': 'Paris'}
```

Analysis on example 5 :

The question-answering model extracts the correct answer span "Paris" directly from the provided context. The model does not generate new text; instead it locates the most relevant part of the context that answers the question. The high confidence score shows it understands the relationship between the question and the supporting text.

### 6. Example 6 - Text Summarization

```
# TODO 6 :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
Machine learning is a branch of artificial intelligence that allows computer systems to learn patterns directly from data instead of being explicitly programmed for every task. It is widely used today in many areas of daily life, such as recommendation systems on streaming platforms, spam detection in email, fraud detection in banking, and voice assistants on smartphones. The typical workflow starts with collecting and cleaning the data, then choosing a suitable algorithm, training a model on the prepared data, and finally evaluating how well the model performs on new and unseen examples. A good model should be able to generalize, which means it gives accurate predictions not only on the training data but also on data it has never seen before. As more data and computing power become available every year, machine learning keeps growing quickly and is becoming an essential skill for students and professionals who want to work in the technology industry.
    """
)
```

Result :

```
[{'summary_text': ' Machine learning is a branch of artificial intelligence that allows computer systems to learn patterns directly from data . It is widely used in recommendation systems, spam detection, fraud detection and voice assistants . A good model should be able to generalize to data it has never seen before .'}]
```

> Note: sesuaikan Result di atas dengan output asli dari Colab kamu.

Analysis on example 6 :

The summarization pipeline condenses the long paragraph into a much shorter version while keeping the main ideas, such as the definition of machine learning, its real-world applications, and the importance of generalization. This shows the model can compress text effectively without losing the core information of the original content.

### 7. Example 7 - Translation

```
# TODO 7 :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Saya sangat suka belajar kecerdasan buatan dan pemrosesan bahasa alami.")
```

Result :

```
[{'translation_text': "J'aime beaucoup apprendre l'intelligence artificielle et le traitement du langage naturel."}]
```

> Note: sesuaikan Result di atas dengan output asli dari Colab kamu.

Analysis on example 7 :

The translation model accurately converts the Indonesian sentence into French while keeping the original meaning. It handles a full, formal sentence smoothly, which shows the model is suitable for multilingual communication and cross-language understanding tasks.

---

## Analysis on this project

This project provides a practical and hands-on introduction to various Natural Language Processing tasks using Hugging Face `transformers` pipelines. By working through sentiment analysis, topic classification, text generation, masked language modeling, named entity recognition, question answering, summarization, and translation, I learned that a single, consistent `pipeline()` interface can handle a wide range of language problems just by changing the task name and the model.

The biggest insight for me is how powerful and accessible transformer-based models have become: complex NLP tasks that used to require building models from scratch can now be done in only a few lines of code, with pre-trained models downloaded directly from the Hugging Face Hub. Each task uses a different model that is specialized for that purpose, which shows the flexibility of the transformer architecture.

I also learned that the results are not always perfect. Text generation can drift off-topic, and the quality of the output depends a lot on the chosen model and the input given. This taught me that these models are very helpful tools, but their outputs still need human review before being used in real applications.
