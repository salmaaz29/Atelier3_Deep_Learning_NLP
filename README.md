# Atelier 3 --- Deep Learning et NLP sur des textes arabes

## 1. Présentation 

Application du Deep Learning au traitement
automatique du langage naturel (NLP), à partir d'articles rédigés en
arabe autour du thème **« Applications et impacts de l'intelligence
artificielle »**.

Le travail est organisé autour de deux objectifs complémentaires :

1.  **Classification de textes** : prédire le niveau de pertinence d'un
    article par rapport au thème étudié.
2.  **Modélisation du langage** : adapter un modèle pré-entraîné AraGPT2
    à notre corpus, puis générer du texte en arabe à partir de prompts.

## 2. Objectifs

-   Constituer et préparer un corpus d'articles arabes.
-   Nettoyer et prétraiter les textes.
-   Comparer plusieurs architectures récurrentes pour la classification.
-   Étudier l'effet de différentes représentations linguistiques :
    tokens, stemming et lemmatisation.
-   Réaliser le fine-tuning d'AraGPT2.
-   Évaluer les modèles et analyser leurs limites.

## 3. Données et grille de notation

Les articles ont été collectés à partir de sources d'information en
ligne. Chaque article a reçu un score manuel de pertinence compris entre
0 et 10.

La conversion des scores en classes est la suivante :

    Score manuel Classe
  -------------- -----------------
           0 à 3 `peu_pertinent`
           4 à 6 `moyen`
          7 à 10 `pertinent`

La notation manuelle permet de construire les étiquettes utilisées pour
entraîner les modèles de classification. Elle peut toutefois comporter
une part de subjectivité.

> **Sources des articles :** les URL des articles collectés sont
> conservées dans le fichier `dataset_brut.csv`. Les noms des sites et
> les URL doivent être consultés dans ce fichier.

## 4. Pipeline de traitement

Le pipeline comprend les étapes suivantes :

1.  **Collecte** : récupération des articles et de leurs informations
    (site, URL, texte et score).
2.  **Nettoyage** : suppression du bruit et des éléments inutiles.
3.  **Prétraitement linguistique** : normalisation du texte arabe,
    suppression des mots vides, stemming et lemmatisation.
4.  **Étiquetage** : transformation des scores manuels en trois classes.
5.  **Représentation numérique** : transformation des textes en
    séquences d'indices et utilisation d'embeddings.
6.  **Séparation des données** : division en ensembles d'entraînement,
    de validation et de test au niveau des articles.
7.  **Entraînement et évaluation** : comparaison des architectures et
    analyse des résultats.

## 5. Classification des articles

Plusieurs architectures récurrentes ont été étudiées avec **PyTorch** :

-   RNN
-   BiRNN
-   GRU
-   LSTM

Les expériences ont également comparé différentes variantes de
prétraitement :

-   `tokens` : tokens après le prétraitement de base ;
-   `tokens_stem` : tokens après stemming ;
-   `tokens_lemma` : tokens après lemmatisation.

Les modèles sont évalués à l'aide de métriques de classification,
notamment l'accuracy, la précision, le rappel et le F1-score. Les
matrices de confusion permettent d'examiner les erreurs par classe.

### Résultats principaux

Dans les expériences réalisées :

-   Le **LSTM** a obtenu le meilleur F1-score macro sur l'ensemble de
    validation : **0,823**.
-   Le **GRU** a obtenu les meilleurs résultats sur l'ensemble de test
    parmi les modèles comparés : **accuracy = 0,658** et **F1-score
    macro = 0,617**.

La différence entre les résultats de validation et de test rappelle
qu'un modèle performant sur la validation ne généralise pas
nécessairement aussi bien sur des données jamais utilisées pendant le
réglage.

Les résultats détaillés des expériences sont à consulter dans les
fichiers `resultats_*.csv` lorsqu'ils sont présents dans le dépôt.

## 6. Modélisation du langage et fine-tuning d'AraGPT2

La deuxième partie du projet est consacrée à la génération de texte.
Contrairement à la classification, qui prédit une classe pour un
article, un modèle de langue apprend à prédire les tokens suivants afin
de produire une séquence de texte.

Le modèle utilisé est :

-   **Modèle pré-entraîné :** `aubmindlab/aragpt2-base`
-   **Tâche :** causal language modeling
-   **Corpus de fine-tuning :** articles jugés pertinents (score
    supérieur ou égal à 7)
-   **Nombre d'articles d'entraînement :** 14
-   **Nombre d'articles d'évaluation regroupés :** 6
-   **Taille des blocs :** 256 tokens
-   **Batch size :** 4
-   **Nombre d'époques dans l'expérience finale présentée :** 5
-   **Learning rate :** `5e-5`

Le fine-tuning a été réalisé avec la bibliothèque Hugging Face
Transformers et PyTorch.

### Résultats du fine-tuning

  Mesure         Avant fine-tuning   Après 5 époques
  ------------ ------------------- -----------------
  Perplexité                 315,6              93,5
  BLEU-2                     0,007             0,021

La loss moyenne d'entraînement a diminué au fil des époques :

    Époque   Loss moyenne
  -------- --------------
         1          7,546
         2          5,435
         3          4,869
         4          4,608
         5          4,478

La baisse de la perplexité indique que le modèle s'est mieux adapté au
corpus. Le BLEU-2 a également augmenté par rapport au modèle initial.
Toutefois, sa valeur reste faible et il a légèrement diminué par rapport
à l'expérience précédente de trois époques (0,024). Ces métriques
doivent donc être interprétées avec prudence.

### Exemples de génération

Les prompts suivants ont été utilisés pour tester la génération en arabe
:

**Prompt 1**

`الذكاء الاصطناعي في الطب`

Le modèle produit un texte lié à l'intelligence artificielle en médecine
et cite plusieurs maladies et domaines médicaux. Le texte reste
globalement dans le thème, mais comporte des répétitions et certaines
formulations peu cohérentes.

**Prompt 2**

`يعتمد الذكاء الاصطناعي على`

La génération contient du vocabulaire lié aux algorithmes, aux modèles
et aux données. Cependant, certaines phrases sont répétitives ou
grammaticalement incorrectes.

**Prompt 3**

`تطبيقات الذكاء الاصطناعي في التعليم`

Le modèle évoque les étudiants, les plateformes éducatives,
l'apprentissage automatique et plusieurs disciplines scientifiques. Le
contenu est en rapport avec le prompt, mais présente des répétitions et
des erreurs de formulation.

**Bilan qualitatif :** AraGPT2 parvient à générer des textes en rapport
avec les sujets proposés. Néanmoins, la cohérence, la fluidité et la
correction grammaticale des textes générés restent perfectibles.

## 7. Limites

Plusieurs limites ont été observées :

-   La taille réduite du dataset limite la capacité des modèles à
    apprendre des représentations généralisables.
-   La notation manuelle des articles peut être subjective.
-   Les performances peuvent varier selon la classe et la séparation
    entraînement/validation/test.
-   Le corpus utilisé pour le fine-tuning d'AraGPT2 est limité à 14
    articles pertinents pour l'entraînement.
-   La diminution de la loss d'entraînement ne garantit pas, à elle
    seule, une bonne généralisation.
-   Le BLEU-2 mesure le chevauchement lexical avec une référence, mais
    ne représente pas entièrement la qualité sémantique ou grammaticale
    d'un texte généré.

## 8. Ce que j'ai appris

Cet atelier m'a permis de mettre en pratique plusieurs notions de Deep
Learning et de NLP :

-   Collecter et nettoyer des textes arabes : normalisation, suppression
    des mots vides, stemming et lemmatisation.
-   Transformer les textes en séquences d'indices et utiliser des
    embeddings.
-   Construire et entraîner des modèles RNN, BiRNN, GRU et LSTM avec
    PyTorch.
-   Comprendre la différence entre classification de textes et
    modélisation du langage.
-   Effectuer le fine-tuning d'un Transformer pré-entraîné avec Hugging
    Face.
-   Évaluer des modèles avec des métriques adaptées à la classification
    et à la génération.

## 10. Structure indicative du dépôt

``` text
.
├── Atelier3_DL_NLP.ipynb
├── dataset_brut.csv
├── matrices_confusion.png
├── courbes_loss.png
├── resultats_*.csv
└── README.md
```


