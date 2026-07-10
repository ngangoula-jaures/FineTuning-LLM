# Fine-tuning Llama-2-7b via QLoRA (Projet Guanaco)

## 🎯 Objectif du projet

Ce projet a pour but de spécialiser un modèle d'intelligence artificielle généraliste (Llama-2, développé par Meta) pour mieux répondre aux questions de type "instruction".

Imaginez que vous ayez un assistant très cultivé mais qui ne connaît pas forcément les règles de politesse ou le format spécifique que vous attendez. Le "fine-tuning" (ou ajustement fin) est la méthode qui consiste à donner une formation complémentaire à cet assistant sur un jeu de données spécifique (Guanaco) pour qu'il devienne plus efficace, plus précis et mieux adapté à vos besoins précis.

Ici, j’ai utilisé une technique d'optimisation appelée QLoRA qui permet de réaliser cet apprentissage avec des moyens informatiques réduits, rendant cette technologie accessible (Colab GPU T4).

---

## 📁 Structure du Projet

Le projet est composé des fichiers principaux suivants :
* **`FineTuning.ipynb`** : Le notebook principal détaillant tout le pipeline de préparation, de quantification en 4-bit, de configuration de LoRA et l'entraînement du modèle sur le dataset `mlabonne/guanaco-llama2-1k`.
* **`test1_finetuned_model.ipynb`** : Un notebook dédié aux tests comparatifs entre le modèle de base (Llama-2-7b brut) et le modèle fine-tuné (`JauresN16/llama-2-7b-custom-finetune`). Il met en évidence le net changement de comportement suite à l'apprentissage.

---

## 🚀 Architecture Technique

* **Modèle de base :** `meta-llama/Llama-2-7b-hf` (quantifié en 4-bit via `bitsandbytes`).
* **Méthode d'adaptation :** LoRA (Low-Rank Adaptation) avec `peft` (r=64, alpha=16, dropout=0.1).
* **Optimisation mémoire :**
  * `gradient_accumulation_steps=4`.
  * `paged_adamw_32bit` pour la gestion optimisée des gradients.
  * Désactivation du KV Cache (`use_cache=False`) pour l'entraînement.
* **Stratégie d'entraînement :**
  * Validation split (90% Train / 10% Eval).
  * `EarlyStoppingCallback` pour prévenir le sur-apprentissage (overfitting).
  * Évaluation automatique à chaque étape de sauvegarde.

---

## 🧪 Tests et Évolution du Comportement (Base vs Fine-tuné)

Afin de valider l'impact du fine-tuning, un test comparatif a été mis en place dans le fichier `test1_finetuned_model.ipynb`. Les résultats illustrent parfaitement la différence de paradigme entre un modèle de base et un modèle instruct :

### 1. Modèle de Base (`meta-llama/Llama-2-7b-hf`)
* **Comportement constaté :** Complétion de texte brute.
* **Analyse :** Le modèle d'origine ne cherche pas à *répondre* à l'instruction. Face à un prompt comme *"What is backpropagation?"*, il se contente de calculer la probabilité des mots suivants. Cela résulte souvent en la génération de questions similaires, de répétitions, ou de suites de phrases dénuées d'intention conversationnelle.

### 2. Modèle Fine-tuné (`JauresN16/llama-2-7b-custom-finetune`)
* **Comportement constaté :** Amorce du suivi d'instructions et comportement conversationnel.
* **Analyse :** Bien qu'entraîné sur un nombre réduit d'étapes (contraintes matérielles sur Colab), le modèle fine-tuné abandonne la simple complétion. Il adopte un ton plus naturel et direct (*"For example, let us consider..."*), s'adressant à l'utilisateur pour formuler une véritable réponse à l'instruction donnée.

---

## 🛠 Défis rencontrés et Solutions

* **Gestion de la mémoire VRAM :** Implémentation du QLoRA pour réduire l'empreinte mémoire, permettant le fine-tuning d'un modèle 7B sur un GPU grand public.
* **Dégénérescence de la génération :** Résolution des boucles de répétition de texte par l'ajustement dynamique des hyperparamètres d'inférence (`repetition_penalty`, `temperature`, `top_p`).
* **Mise à jour des bibliothèques :** Migration fluide vers `SFTConfig` et `processing_class` pour assurer la compatibilité avec les dernières versions de la stack `trl` et `transformers`.

---

## 💻 Usage

Pour charger et tester le modèle fine-tuné (disponible sur le hub Hugging Face) :

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "JauresN16/llama-2-7b-custom-finetune"

# Chargement du modèle et du tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Création du pipeline de génération
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    max_length=200,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
    do_sample=True
)

# Test
prompt = "What is supervised learning?"
result = pipe(prompt)

print(result[0]['generated_text'])
```

---

## 📈 Perspectives

* Passage à un dataset de plus grande taille pour améliorer la généralisation et la profondeur des réponses.
* Test avec d'autres architectures (comme Mistral 7B) en utilisant le même pipeline de fine-tuning.