# Projeto de Processamento de Linguagem Natural com Transformers

## Informações:

* Andreza Maria Coutinho Falcão - andreza.mcfalcao@ufrpe.br
* Beatriz Pereira da Silva - beatriz.pereiras@ufrpe.br
* Leonardo da Silva Viana Filho - leonardo.vianafilho@ufrpe.br
* Victor Schimitz Donato Cavalcanti - victor.schmitz@ufrpe.br

## Objetivo

Este projeto demonstra como utilizar a biblioteca `transformers` para realizar várias tarefas de Processamento de Linguagem Natural (NLP) com modelos pré-treinados. As tarefas abordadas incluem:

1. Análise de Sentimento
2. Classificação de Texto
3. Classificação Zero-Shot
4. Resumo de Texto
5. Perguntas e Respostas (Question Answering)

## Requisitos

- Python 3.7 ou superior
- Biblioteca `transformers`
- Biblioteca `torch`

Você pode instalar as bibliotecas necessárias usando pip:

```bash
pip install transformers torch
```

## Descrição

Este projeto utiliza a biblioteca `transformers` para aplicar modelos pré-treinados em tarefas específicas de NLP. A biblioteca `transformers` da Hugging Face fornece uma interface fácil para diversos modelos de linguagem, como BERT, DistilBERT, BART, e mais.

### Tarefas Demonstradas

1. **Análise de Sentimento:**
   Utiliza um modelo pré-treinado para determinar o sentimento (positivo ou negativo) de um texto.

2. **Classificação de Texto:**
   Usa um modelo pré-treinado para classificar textos em categorias específicas.

3. **Classificação Zero-Shot:**
   Aplica uma abordagem de zero-shot para classificar textos em categorias não vistas durante o treinamento.

4. **Resumo de Texto:**
   Gera um resumo para um texto longo usando um modelo de sumarização.

5. **Perguntas e Respostas:**
   Responde perguntas com base em um contexto fornecido usando um modelo de QA.

## Passo a Passo

### 1. Análise de Sentimento

```python
from transformers import pipeline

# Cria um pipeline de análise de sentimento
classifier = pipeline("sentiment-analysis")

# Aplica o pipeline a um texto
result = classifier("I loved Star Wars so much!")
print(result)
```

**Descrição:** Utiliza um modelo pré-treinado para classificar o sentimento do texto como positivo ou negativo.

### 2. Classificação de Texto

```python
from transformers import pipeline

# Cria um pipeline de classificação de texto com um modelo BERT
classifier_txt = pipeline("text-classification", model="bert-base-uncased")

# Textos para classificação
texts_classif = [
    "I love this movie! It's fantastic.",
    "I hate this movie. It's terrible."
]

# Obtém os resultados da classificação
results = classifier_txt(texts_classif)
print(results)
```

**Descrição:** Classifica os textos em categorias específicas usando um modelo pré-treinado BERT.

### 3. Classificação Zero-Shot

```python
from transformers import pipeline

# Cria um pipeline de classificação zero-shot com um modelo BART
classifier_multi = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Textos para classificação
texts_classif = [
    "I love this movie! It's fantastic.",
    "The political situation is very tense.",
    "The new technology in AI is groundbreaking.",
    "The recent sports event was thrilling."
]

# Labels (categorias) que você deseja classificar
candidate_labels = ["entertainment", "politics", "technology", "sports"]

# Aplica a classificação zero-shot
results = [classifier_multi(text, candidate_labels) for text in texts_classif]

# Imprime os resultados
for result in results:
    print(result)
```

**Descrição:** Classifica textos em categorias não vistas durante o treinamento usando uma abordagem zero-shot com BART.

### 4. Resumo de Texto

```python
from transformers import pipeline

# Cria um pipeline de sumarização
sumarizacao = pipeline("summarization")

# Texto para sumarização
text = "Transformer models are a type of deep neural network, similar to other types of neural networks such as Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs). The core of Transformer models is the self-attention mechanism, which allows the model to view different parts of a sequence simultaneously and determine the importance of each part. To understand the self-attention mechanism more clearly, imagine you are in a noisy room trying to hear a specific voice. Your brain can automatically focus on that specific voice while trying to ignore other voices. The self-attention mechanism works in a similar way."

# Obtém o resumo
result = sumarizacao(text)
print(result)
```

**Descrição:** Gera um resumo de um texto longo utilizando um modelo de sumarização.

### 5. Perguntas e Respostas

```python
from transformers import pipeline

# Cria um pipeline de perguntas e respostas
question_answerer = pipeline("question-answering")

# Contexto e pergunta
context = "🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration between them."
question = "Which deep learning libraries back 🤗 Transformers?"

# Obtém a resposta
result = question_answerer(question=question, context=context)
print(result)
```

**Descrição:** Responde a perguntas com base em um contexto fornecido usando um modelo de QA.

## Referências

- **Hugging Face Transformers:** A biblioteca `transformers` fornece uma interface fácil para diversos modelos de linguagem. Para mais informações, consulte a [documentação oficial](https://huggingface.co/transformers/).
  
- **Paper:** "Attention is All You Need" (2017) - Introduz a arquitetura Transformer, que revolucionou o campo do Processamento de Linguagem Natural. Você pode ler o paper completo [aqui](https://arxiv.org/abs/1706.03762).
