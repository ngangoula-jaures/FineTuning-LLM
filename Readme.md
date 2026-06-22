# Fine-tuning Llama-2-7b via QLoRA (Projet Guanaco)

## 🎯 Objectif du projet

Ce projet a pour but de spécialiser un modèle d'intelligence artificielle généraliste (Llama-2, développé par Meta) pour mieux répondre aux questions de type "instruction".

Imaginez que vous ayez un assistant très cultivé mais qui ne connaît pas forcément les règles de politesse ou le format spécifique que vous attendez. Le "fine-tuning" (ou ajustement fin) est la méthode qui consiste à donner une formation complémentaire à cet assistant sur un jeu de données spécifique (Guanaco) pour qu'il devienne plus efficace, plus précis et mieux adapté à vos besoins précis.

Ici, j’ai utilisé une technique d'optimisation appelée QLoRA qui permet de réaliser cet apprentissage avec des moyens informatiques réduits, rendant cette technologie accessible (Colab GPU T4).

---

## 🚀 Architecture Technique

* **Modèle de base :** Llama-2-7b (quantifié en 4-bit via bitsandbytes).
* **Méthode d'adaptation :** LoRA (Low-Rank Adaptation) avec peft.
* **Optimisation mémoire :**
  * gradient_accumulation_steps=4.
  * paged_adamw_32bit pour la gestion optimisée des gradients.
  * Désactivation du KV Cache (use_cache=False) pour l'entraînement.
* **Stratégie d'entraînement :**
  * Validation split (90% Train / 10% Eval).
  * EarlyStoppingCallback pour prévenir le sur-apprentissage (overfitting).
  * Évaluation automatique à chaque étape de sauvegarde.

---

## 🛠 Défis rencontrés et Solutions

* **Gestion de la mémoire VRAM :** Implémentation du QLoRA pour réduire l'empreinte mémoire, permettant le fine-tuning d'un modèle 7B sur un GPU grand public.
* **Dégénérescence de la génération :** Résolution des boucles de répétition de texte ([[[[) par l'ajustement dynamique des hyperparamètres d'inférence (repetition_penalty, temperature, top_p).
* **Mise à jour des bibliothèques :** Migration fluide vers SFTConfig et processing_class pour assurer la compatibilité avec les dernières versions de la stack trl et transformers.

---

## 📊 Performance

Le modèle démontre une convergence stable avec une décroissance constante de la Training Loss et de la Validation Loss, validant la capacité du modèle à apprendre la structure des instructions Guanaco.

---

## 💻 Usage

Pour charger et utiliser ce modèle :

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
model = PeftModel.from_pretrained(model, "JauresN16/llama-2-7b-finetune")
```

---

## 📈 Perspectives

* Passage à un dataset de plus grande taille pour améliorer la généralisation.
* Test avec d'autres architectures (Mistral 7B) en utilisant le même pipeline de fine-tuning.