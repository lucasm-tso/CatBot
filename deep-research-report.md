# Extraction de données structurées à partir de PDF : état de l’art et guide pratique

## Résumé exécutif  
Les documents PDF représentent un défi majeur pour l’extraction automatique de données structurées (texte, tables, formulaires, etc.) car ils combinent contenus numériques et images scannées. Cet état de l’art présente un panorama des outils (open-source et commerciaux), des algorithmes et pipelines pour convertir des PDF en données prêtes à charger en base. On y compare notamment Tesseract, EasyOCR, PaddleOCR, Tabula/Camelot, AWS Textract, Azure Form Recognizer, ABBYY, etc., avec leurs langages supportés, licences, forces/faiblesses et coûts. Les approches mêlent méthodes à base de règles (heuristiques PDF natives) et modèles ML/DL (LayoutLM, Donut, Table Transformer, réseaux CNN/RNN pour OCR), avec étapes d’analyse de mise en page, détection de tables, reconnaissance de texte et post-traitement. Un pipeline typique intègre prétraitement d’image (redressement, débruitage), OCR (si besoin), détection de zones (texte, tables, champs), extraction de structure tabulaire, reconnaissance d’entités par NLP (SpaCy, BERT, etc.), normalisation des données, et validation manuelle. La qualité se mesure en taux de reconnaissance (CER/WER pour texte), précision/recall de détection de tables et F1-score pour l’extraction d’entités. On dispose de jeux de données et benchmarks publics (PubTables-1M, ICDAR, FUNSD, SROIE, etc.) pour évaluer et comparer les solutions. Enfin, des considérations pratiques (scalabilité CPU/GPU, conteneurs Docker, respect GDPR, journalisation d’erreurs) et l’intégration dans des bases SQL/NoSQL/graph sont discutées. Le rapport propose un tableau comparatif coûts/licences et un plan de prototypage par étapes (collecte échantillon, test d’outils, itérations d’amélioration), avec code d’exemple et schéma pipeline recommandé.  

## Outils et bibliothèques  

- **Apache PDFBox (Java, Apache 2.0)** – Bibliothèque Java robuste pour manipuler et extraire du PDF (texte Unicode, fusion/division, impression, conversion en image, gestion des formulaires)【28†L55-L60】. Idéal pour PDF natifs : très précis, licence libre, mais ne gère pas l’OCR sur image.  
- **PyMuPDF / MuPDF (Python+C, AGPL)** – Extraction rapide de texte et d’images de PDF (UTF-8)【31†L44-L49】. Outil très performant CPU-only (10× plus rapide que concurrent selon développeurs). Code source AGPL (exige ouverture), avec options commerciales pour fonctionnalités avancées (analyse de mise en page). Support natif d’annotations et d’extraction de structure, sans nécessiter OCR pour texte numérique.  
- **pdfminer.six + pdfplumber (Python, MIT)** – pdfminer.six (MIT) analyse la structure des PDF et extrait le texte, mais sans outils tables【37†L93-L100】. pdfplumber (MIT) construit dessus : il extrait tous les caractères, lignes, rectangles et peut détecter des tableaux simples (à base texte)【37†L93-L100】. Fonctionne très bien sur PDF *machine-generated* avec couche texte propre【37†L93-L100】, peu adapté aux scans ou formulaires complexes.  
- **Tabula (Java/MIT)** – Outil gratuit (interface web) pour extraire des tableaux de PDF texte. Développé par la communauté, licence MIT【33†L342-L345】. Efficace sur tables « propres » et multi-pages【10†L93-L100】【33†L342-L345】, mais ne gère pas les scans manuscrits ni documents hétérogènes (images)【10†L115-L123】. Dispose aussi d’une bibliothèque Java et CLI (tabula-java) pour traitement batch.  
- **Camelot (Python, MIT)** – Bibliothèque Python pour l’extraction de tables PDF (texte). Licence MIT【35†L75-L83】. Permet de configurer finement l’extraction (mode *lattice* ou *stream*), retourne les tables sous forme de DataFrame Pandas (export CSV/JSON/SQL)【35†L154-L160】【35†L75-L83】. Convient à PDF machine-generated sans recourir à OCR. Ne supporte pas les documents scannés.  
- **OCR Tesseract (C++, Apache 2.0)** – Moteur OCR open-source de Google. Reconnaissance robuste de texte imprimé (182 langues, y compris français). A partir de la version 4, intègre des réseaux LSTM pour améliorer l’extraction【61†L226-L233】. Très utilisé (p. ex. via python-tesseract) pour numériser des scans. Résultats variables selon qualité de l’image ; peut être combiné à des bibliothèques de prétraitement d’image (OpenCV) et à des correcteurs post-OCR.  
- **EasyOCR (Python, Apache 2.0)** – Bibliothèque OCR multi-langues (80+ langues et scripts latins, arabes, cyrilliques, etc.)【43†L519-L523】【43†L503-L512】. Basée sur PyTorch (CRNN+CTC) avec modèle de détection CRAFT. Licence Apache 2.0. Bonne précision hors maniement de texte manuscrit fin, et configuration clé-en-main.**  
- **docTR (Python, Apache 2.0)** – Outil OCR deep learning (Mindee) en open source【41†L327-L335】【41†L301-L304】. Pipeline end-to-end (détection de mots + reconnaissance de caractères) configurable. Exploite des modèles CNN/Torch pré-entraînés. Utile pour PDF ou images complexes ; se connecte directement à PDF et à URL (ex. *DocumentFile.from_pdf()*).  
- **PaddleOCR (Python, Apache 2.0)** – Framework OCR/Document AI (PaddlePaddle) très complet. Supporte >100 langues et manuscrit de façon limitée【45†L377-L385】【8†L169-L177】. Dispose de modèles SOTA (ex. PaddleOCR-VL-1.5, 0,9B paramètres) pré-entraînés pour détection de texte, d’images, de tables, de zones complexes【45†L429-L437】. Offre aussi une suite pour l’extraction d’information (PP-StructureV3 convertit PDF en Markdown/JSON)【45†L439-L447】. Gratuit, open source, mais construction de pipeline plus lourde.  
- **LayoutParser (Python, Apache 2.0)** – Bibliothèque multimodale (par *ai.baidu.com*) qui fournit des modèles de segmentation de document (détectant blocs de texte, titres, tables, figures) pré-entraînés (souvent sur Detectron2). Permet d’identifier la mise en page avant OCR/NLP. Par exemple, un usage typique est **: segmenter l’image en zones textuelles et tabulaires avant extraction.

- **Amazon Textract (Cloud AWS, service payant)** – Service ML d’extraction intelligent de documents (OCR amélioré)【56†L58-L67】. Extrait texte imprimé et manuscrit, ainsi que structures (tables, champs de formulaires) avec un appel API. Très précis sur documents propres et mixtes (scans, photos)【9†L312-L320】【56†L58-L67】. Simplicité d’intégration via AWS SDK (ex. bibliothèque *Textractor*). Coût typique: env. **\$15** par 1 000 pages pour le premier million (puis \$30/1K pages avec fine-tuning)【9†L300-L308】.  
- **Azure Form Recognizer / Document Intelligence (Microsoft, Cloud payant)** – Service comparable à Textract. API REST qui analyse formulaires et tableaux. 10 $ pour 1 000 pages (modèle “layout”)【9†L332-L340】, volume plus important voire conteneurs dédiés pour gros flux. Avantage: docs et outils Azure (incl. modèles préconstruits factures, reçus). Résultats très bons sur divers types de docs (manuscrit compris)【9†L339-L347】.  
- **Google Document AI (Cloud payant)** – Ensemble d’API de Google (form parser, OCR avancé) et interface Pinpoint (beta). Bonnes performances sur tableaux et formulaires complexes【8†L240-L249】. Prix atypique par requête/page (→ ex. \$38 pour ~1 266 pages【57†L1-L8】). Moins prévisible que Textract/Azure en tarifs.  
- **ABBYY (commercial)** – Outils leaders (FineReader, FlexiCapture, Cloud OCR) : très haute précision (>98 %), support de dizaines de langues (français inclus) et reconnaissance de structure (documents en colonnes, zones de texte, champs de formulaires). Forte capacité de réglage (« FlexiLabels », modèles pré-configurés pour factures, etc.). Coût élevé (licences logicielles ou tarification cloud page) justifié pour besoins industriels.  

Chaque outil présente des forces/faiblesses typiques. Par exemple, Tabula/Camelot conviennent aux PDF natifs simples, tandis que PaddleOCR/docTR/EasyOCR sont orientés image-scanners. Les services Cloud (Textract, Azure, Google) offrent le plus d’automatisation (et de performance sur scans complexes) mais impliquent un budget et des enjeux de confidentialité. Les technologies basées sur DL (LayoutLM, Table Transformer, Donut, etc.) demandent plus d’expertise et de puissance de calcul, mais surpassent les approches classiques dans les tâches de lecture complexe (tableaux imbriqués, textes manuscrits)【4†L69-L77】【26†L309-L315】.

## Techniques et algorithmes  

- **Parsing natif de PDF** – Les PDF « digitaux » contiennent souvent un flux de texte Unicode. On peut alors appliquer des règles simples ou des librairies dédiées (PDFBox, PDFMiner) pour extraire le texte avec positionnement. Cette approche donne des résultats parfaits sur du PDF généré (texte interne) mais échoue sur PDF scannés. Des heuristiques (regex, patron de mise en page) permettent de repérer titres, paragraphes ou champs de formulaires **(AcroForms)** sans ML.  
- **OCR et LSTM** – Les moteurs OCR (Tesseract, EasyOCR, PaddleOCR, docTR, etc.) détectent le texte dans des images (pages scannées). Tesseract 5 utilise des LSTM pour améliorer la reconnaissance des séquences de caractères. Ces systèmes retournent pour chaque caractère/mot un taux de confiance. La qualité dépend fortement du prétraitement (binairisation, élimination de bruit, correction de perspective). Le post-traitement inclut souvent l’utilisation de dictionnaires/langage pour corriger les erreurs (“post-ocr denoising”), et la reconstitution de mots coupés (fin de ligne).  
- **Analyse de mise en page (Layout)** – Les outils comme LayoutParser (basés sur Detectron2) ou PaddleOCR-VL segmentent l’image en zones (textes, tables, figures). Les modèles récents comme *LayoutLM* (BERT multimodal de Microsoft) apprennent simultanément texte et layout【16†L69-L77】. LayoutLM (2020) pré-entraîné sur millions de documents considère chaque mot avec sa bbox, ce qui améliore nettement l’extraction d’informations (formulaires, reçus, classification)【16†L69-L77】. Ses successeurs (LayoutLMv2/XLMv3) et dérivés (DocuNet, StructExt…) ajoutent des caractéristiques visuelles ou des graphes de mots. Donut (Document UNderstanding Transformer) pousse plus loin : c’est un transformeur visuel-texte sans OCR intermédiaire, lisant l’image directement (OCR-free) et promettant un gain d’efficacité en évitant la chaîne OCR classique【4†L69-L77】.  
- **Détection de tables et reconnaissance de structure** – On distingue : *détection de la présence de tables* (bounding boxes) et *structure interne* (cellules, lignes/colonnes). Des algorithmes classiques détectent traits (Hough transform) ou zones blanches pour segmenter les cellules. Des modèles DL plus récents (TableNet, CascadeTabNet, Tablet Transformer) formalisent la détection comme un problème d’objets: par ex. Microsoft propose *Table Transformer* (TaT) en s’appuyant sur DETR pour détecter automatiquement tables et leurs cellules en une passe【47†L462-L470】. De nombreux jeux de données annotées (Marmot, ICDAR, PubTabNet) ont permis d’entraîner et de mesurer ces réseaux. La métrique GriTS (Grid table similarity) a été introduite pour évaluer la structure reconnue de chaque cellule【47†L486-L494】.  
- **Reconnaissance de formulaires (Key-Value)** – Après OCR, il faut associer champs-clés et valeurs (p. ex. « Nom : Victor Hugo »). Des approches basiques font du matching positionnel (templates), efficaces sur formulaires fixes. Les méthodes ML (LayoutLM et variantes KIE) combinent texte, position et visuel pour faire de la NER spatiale. Par exemple, PaddleOCR propose un pipeline “Key Information Extraction” inspiré de LayoutXLM pour extraire champs structurés【26†L307-L315】. LayoutLMv3 intègre directement vision+texte pour la tâche de KIE. Des graphes relationnels (GCN) ont été essayés pour capturer les liens champ-valeur. En complément, de l’**entity extraction** classique (SpaCy, camemBERT pour le français, polyglot, etc.) permet d’identifier entités nommées dans le texte extrait (noms propres, dates, montants). Ce NER peut être entraîné sur le domaine (factures, rapports) pour repérer adresses, numéros ID, etc.  
- **Tableau → CSV/BD** – Une fois les cellules d’une table identifiées (position + texte), on reconstruit l’ordre (lignes/colonnes) pour générer un CSV. Des outils comme Camelot et Tabula offrent cette fonctionnalité. Les modèles DL produisent souvent directement une structure arborescente exportable (par ex. Document AI de Google sort du JSON). Il faut traiter les « split » ou fusions de cellules (cellspan) correctement. Enfin, on attribue souvent un score global de confiance à chaque donnée extraite (fondé sur les scores des détecteurs et l’OCR) pour guider la validation manuelle ultérieure.  

## Architecture pipeline recommandée 

Un pipeline complet combine plusieurs étapes :  

1. **Prétraitement** – Si le PDF est numérique, on saute l’OCR pour lire directement le texte via PDFBox, PyMuPDF ou pdfminer. Si c’est une image (scan), on convertit d’abord en images (pp. ex. via PyMuPDF) et on améliore l’image : **binarisation**, **deskew** (redressement des pages penchées), **denoising**.  
2. **Reconnaissance (OCR)** – Sur pages image, lancer l’OCR (par ex. Tesseract ou PaddleOCR) pour obtenir du texte brut et ses coordonnées. Certains systèmes (docTR, EasyOCR) donnent du texte déjà segmenté en mots/paragraphes. Si besoin, effectuer un second tour de correction (reconnaître une fois, comparer avec dictionnaire, ré-exécuter).  
3. **Analyse de mise en page** – Détecter les blocs logiques : paragraphes, titres, images, tableaux. On peut utiliser des modèles de segmentation (LayoutParser ou PaddleOCR-VL) pour générer des bounding boxes classifiées. Par exemple, un modèle CNN (Detectron2) détectera les zones de tableau et de cellule avec leur position.  
4. **Extraction de tables** – Pour chaque zone de type table, lancer un extracteur de table : soit un algorithme classique (détecter traits horizontaux/verticaux), soit un modèle DL (Table Transformer, CascadeTabNet) qui renvoie la structure ligne/colonne. Ensuite, exécuter l’OCR sur chaque cellule (si non déjà fait page global), reconstruire la table et exporter en CSV/JSON.  
5. **Extraction d’entités/NLP** – Appliquer du NLP sur les blocs de texte restants : pipelines de tokenisation + NER (ex. spaCy, ou un modèle LayoutLM affiné sur entités métier). Identifier les champs-clés (dates, numéros, noms, etc.) et normaliser leur format (ex. dates en ISO). Des librairies comme JPO/regex ou dateparser aident ici.  
6. **Validation et correction (Human-in-the-loop)** – Présenter à l’utilisateur un aperçu des données extraites (tables, champs clés) avec leurs scores de confiance. Permettre la correction manuelle des erreurs (typos OCR, alignement de cellules mal faites). Un système d’étiquetage rapide (ex. interface Document AI ou étiqueteur tiers) peut être envisagé.  
7. **Chargement en base** – Formater les données finales (tables, paires clé/valeur, textes libres) dans la structure de la base de données cible (relationnelle ou NoSQL) et insérer. Des scripts ETL ou ORM peuvent faciliter ce transfert.  

```mermaid
graph LR
  A[Documents PDF] -->|Texte interne| B[Parsing PDF natif (PDFBox/PyMuPDF)]
  A -->|Scan/Image| C[Prétraitement d'image (deskew/denoise)]
  C --> D[OCR (Tesseract, PaddleOCR, docTR, ...)]
  D --> E[Analyse de mise en page (LayoutParser, DETR)]
  E --> F[Détection de tables (CascadeTabNet/TableTransformer)]
  F --> G[Extraction structurelle de tables → CSV/JSON]
  E --> H[Extraction de texte/Entités (LayoutLM, spaCy, regex)]
  H --> I[Normalisation & nettoyage des données]
  G --> I
  I --> J[Validation / Correction humaine (UI)]
  J --> K[Chargement dans Base de données (SQL/NoSQL/Graph)]
```

*Figure : Pipeline typique d’extraction structurée. Les rectangles représentent des étapes technologiques clés, de l’entrée (A) jusqu’au chargement (K).* 

La figure ci-dessus illustre un workflow recommandé. On y voit l’alternance de modules open source et services cloud : par exemple, on peut remplacer la partie OCR+Layout par AWS Textract/Azure Document AI (un seul call REST) ou par un outil LLM/vision (Donut ou GPT-4). Dans une stack open source, une combinaison courante est PyMuPDF (lecture PDF) → Tesseract/EasyOCR → LayoutParser/CascadeTabNet → spaCy → export CSV. Du côté commercial, Azure Document Intelligence et AWS Textract proposent des solutions complètes « clé en main », tandis que Google Document AI facture à l’usage les requêtes complexes. 

## Évaluation : métriques et jeux de données  

- **Métriques OCR et détection** – On mesure l’**exactitude OCR** par le Taux d’erreur par caractère (CER) ou mot (WER) sur un corpus annoté. Pour la détection de zones (tables, champs), on utilise précision/recall sur les boîtes englobantes (IOU thresholds). L’extraction de tables se juge plus finement : le métrique GriTS (Grid Table Similarity) évalue la correspondance cellule par cellule (topologie et contenus)【47†L486-L494】. Pour les entités extraites, on calcule un F1-score par type d’entité (comme pour le NER).  
- **Jeux de données** – De nombreuses collections publiques existent. Par ex. **ICDAR Table** (2013, 2019) avec quelques centaines de tables annotées ; **PubTabNet** (2019) et surtout **PubTables-1M** (2022) contiennent presque 1 million de tables issues d’articles scientifiques, avec annotation de structure fine【51†L116-L124】. Pour les formulaires, **FUNSD** (199 formulaires scannés étiquetés, ICDAR’19) vise le Form Understanding (détection de lignes, correspondance question/réponse)【55†L63-L72】. Pour les reçus/invoices, SROIE (vingt-mille factures annotées clés/valeurs) et des compétitions comme “DocAI” de Google fournissent benchmarks. On peut ainsi tester des modèles sur ces sets pour comparer précision OCR, détection de tables et extraction d’informations.  
- **Méthodologie de test** – Idéalement, on définit un jeu de test représentatif des documents réels (mêmes langues, qualité d’image). On évalue l’ensemble du pipeline en confrontant le JSON/CSV produit au ground-truth : on calcule par exemple le taux de données correctement extraites (p. ex. pour un tableau,  pourcentage de cellules exactes). On documente le comportement sur différents types de documents (texte clair vs brut, avec/ sans tableau, manuscrit vs imprimé). Pour le suivi de performance en production, il est utile de suivre les scores OCR moyens et la fréquence de corrections manuelles.

## Considérations d’implémentation  

- **Scalabilité et parallélisation** – Pour traiter de gros volumes (milliers de pages), distribuer le calcul est essentiel. Les bibliothèques Python (pdfplumber, PyMuPDF) peuvent être parallélisées en multiprocess (ex. via `concurrent.futures`). Les traitements *GPU* (OCR deep learning, segmentation) se prêteront à des pipelines batch sur serveurs GPU ou des conteneurs Kubernetes. Au contraire, des tâches I/O-bound (lecture disque, appels API) peuvent tourner massivement sur CPU. On veillera à découper les PDF en pages ou documents plus petits pour la paralellisation.  
- **Langages et conteneurisation** – Python est majoritaire pour ce genre de pipelines (bibliothèques étendues, prototypage rapide). Java/Scala est utilisé pour des solutions plus grosses (ex. extraction PDFBox sur Spark). On utilisera Docker/OCI pour packager les dépendances (OCR, modèles, librairies). Par exemple, un conteneur Tesseract OCR ou EasyOCR avec Python 3. Des orchestrateurs comme Airflow ou Luigi peuvent gérer les flux de travail.  
- **Monitoring et tolérance aux erreurs** – Intégrer de la journalisation (logging) pour chaque étape (pages traitées, succès/échecs, scores de confiance) permettra de détecter les problèmes de qualité. On peut rejeter ou taguer les pages où l’OCR renvoie un score faible. Des métriques en temps réel (combien de pages/min, taux d’erreurs) facilitent le dimensionnement. Implémenter un système de “retry” ou fallback : ex. si la détection DL échoue, retenter avec heuristique basique.  
- **Vie privée et conformité** – Les documents PDF peuvent contenir des données personnelles (GDPR). Si on utilise des services cloud (Textract, Google AI, etc.), il faut chiffrer les communications et vérifier les engagements de confidentialité (ex. AWS, Azure offrent du Data encryption). On peut préférer des solutions locales (Tesseract, PaddleOCR on-premise) pour éviter l’exfiltration de données. Il convient de gérer l’accès aux résultats (logs sécurisés, anonymisation le cas échéant).  

## Intégration et schéma de base de données  

Les données extraites doivent être stockées dans une base adaptée :  

- **Base relationnelle (SQL)** : chaque table extraite peut devenir une table SQL. Par exemple, avoir une table “doc_table” avec colonnes *_doc_id, table_id, row, col, valeur_*. Les paires clé-valeur extraites d’un formulaire peuvent aller dans une table “metadonnees” (clé, valeur, id_document). Les textes libres peuvent être indexés (full-text). L’avantage est la normalisation et la consistance (clés/types fixes).  
- **NoSQL (document-store)** : on peut stocker le JSON complet de chaque document dans MongoDB ou CouchDB. Utile si la structure varie beaucoup. Par exemple, un document peut contenir un tableau, un formulaire, un paragraphe, chacun comme sous-document JSON.  
- **Graph DB** : si l’on souhaite lier les données extraites entres elles (ex. une entité trouvée dans plusieurs docs ou lier facture-client-commande), un graphe (Neo4j) est indiqué. Par exemple, créer des nœuds pour les entités (personne, montant, référence) extraites et relier via des relations (émis_par, contenu_dans).  
- **Schéma recommandé** : en général, avoir une table centrale “documents” (métadonnées: titre, date, chemin, etc.) et des tables enfants pour les éléments structurés. Par exemple, “document_tables” liste les tables par document, “table_cells” stocke chaque cellule avec sa position. On notera la dimension de normalisation (par ex. ne pas dupliquer les clés de formulaire). Un design clair (ex. *3NF* pour SQL) facilite l’interrogation ultérieure.  

## Coût et licences (comparatif)  

| Outil/Solution                 | Type         | Lang/Plateforme        | Licence / Modèle     | Coût (exemple)                    | Points forts / usages          |
|:-------------------------------|:------------:|:----------------------:|:--------------------:|:---------------------------------:|:------------------------------|
| **Tesseract**                  | OCR         | C++/multi          | Apache 2.0 (Libre)   | Gratuit (open-source)             | Nombreux langages, modifiable, dépendant de la qualité d’image |
| **EasyOCR**                    | OCR         | Python (PyTorch)      | Apache 2.0          | Gratuit                          | +80 langues, prêt-à-l’emploi   |
| **docTR (Mindee)**             | OCR pipeline| Python (PyTorch)      | Apache 2.0          | Gratuit                          | Détection+reco. intégrés      |
| **PaddleOCR**                  | OCR/DocAI   | Python (PaddlePaddle) | Apache 2.0          | Gratuit (open-source)            | OCR multi-langue, structure complexe (tables, formulaires)【45†L429-L437】 |
| **Camelot / Tabula / pdfplumber** | Parsing PDF | Python / Java        | MIT                | Gratuit                          | Extraction de tables (PDF texte)【10†L93-L100】【35†L75-L83】  |
| **Apache PDFBox**              | Parsing PDF | Java                  | Apache 2.0          | Gratuit                          | Texte, formulaires PDF natifs【28†L55-L60】 |
| **AWS Textract**               | OCR/ID      | Service Web           | Propriétaire (AWS)  | ~15 $/1 000 pages (1e 1M)【9†L300-L308】 | OCR + formulaires + tables automatique (OCR+IA)【56†L58-L67】 |
| **Azure Document Intelligence**| OCR/ID      | Service Web           | Propriétaire (MS)   | ~10 $/1 000 pages (Layout)【9†L332-L340】 | OCR + Form Recognizer (bon sur scan/mixte)【9†L332-L340】 |
| **Google Document AI**         | OCR/ID      | Service Web           | Propriétaire (Google)| tarification API (ex. ~\$30 pour ~1 000 pages)【57†L1-L8】 | Solutions GA²L dédiées (facture, KVP, vision par langage) |
| **ABBYY FineReader/Flexi**     | OCR/ID      | Desktop / Serveur     | Propriétaire        | Licence élevée (desktop+serveur) | Très haute précision, multilingue, templates commerciaux |
| **LayoutLM, Donut, TableTransformer** | ML/DL (modeles) | Python/TF/PyTorch | Recherche (licences ouvertes) | Coût GPU & infra           | Très performants sur formulaires, tables ; nécessite données & GPUs【16†L69-L77】【4†L69-L77】 |

Ce tableau compare en synthèse les principales solutions. On voit par exemple que les outils open-source (licences Apache/MIT) sont gratuits mais demandent un développement intégré et offrent une flexibilité totale (pas de contrainte d’usage commercial). Les services cloud facturent à l’usage : AWS Textract est simple mais peut devenir cher en volume【9†L300-L308】; Azure est légèrement moins cher pour du traitement brut【9†L332-L340】. Les modèles de recherche (LayoutLM, Donut) ne coûtent pas de licence logicielle mais impliquent du temps de R&D, du training (GPU), et sont généralement utilisés via frameworks payants (cloud ou on-prem avec licences).  

## Choix de la solution : matrice de décision  

Pour choisir la solution adéquate, on doit pondérer plusieurs facteurs :  

- **Volume de documents** : pour des volumes très élevés, les solutions open-source (Tesseract+Camelot, PyMuPDF) ou le cloud avec forfaits échelonnés sont préférables. Pour de faibles volumes, un outil commercial complet (ABBYY, Azure) peut être amorti par le temps gagné.  
- **Précision requise** : si des données critiques doivent être extraites (contrats légaux, résultats financiers), privilégier les solutions premium (ABBYY, Textract/Azure) ou les modèles DL avancés (LayoutLM + fine-tuning), quitte à superviser les résultats. Si des erreurs sont tolérables, un pipeline open-source peut suffire.  
- **Budget disponible** : les outils open-source n’ont pas de coût licite mais exigent du développement (coût humain). Les services cloud ont un coût par page (pouvant être élevé si volume massif). On établira un calcul coût par page vs exigence qualité pour trouver le seuil de rentabilité.  
- **Nature des documents** : documents « propres » (invoices digitales, PDF natifs) → solutions PDF-parsing (Tabula, PDFBox, PDFPlumber). Documents scannés/mélange de formats → solutions OCR+IA (Textract, PaddleOCR, etc.). Documents manuscrits → peu d’options gratuites (Textract/Azure gèrent partiellement le manuscrit).  
- **Confidentialité** : si les documents contiennent des données sensibles, un système on-premise (Tesseract/PyMuPDF) ou un cloud certifié GDPR (Azure, AWS dans certains datacenters) sera requis.  

En croisant ces critères, on construit une matrice de décision. Par exemple :  
- Cas **volume faible, haute précision** : ABBYY ou Azure Form Recognizer avec interfaces graphiques, validations par l’humain.  
- Cas **volume élevé, budget limité** : pipeline open-source (PyMuPDF → OCR Tesseract/EasyOCR → tables Camelot/Pandas) sur GPU cloud ou serveurs dédiés.  
- Cas **documents variés (multi-langues, image+texte)** : solution DL moderne (PaddleOCR-VL, Donut) ou service cloud généraliste (Textract) pour réduire le travail d’intégration.  

## Plan de prototypage proposé  

**Étape 1 : collecte et analyse d’échantillon**. Rassembler un sous-ensemble représentatif de documents (quelques dizaines). Examiner leur type (PDF natifs vs scans, langues, formats). Servira à tester plusieurs outils.  

**Étape 2 : extraction de base**. Écrire un script rapide (ex. Python) pour extraire le texte : si PDF natifs, via PyMuPDF；sinon avec Tesseract. Mesurer le taux brut d’extraction et la qualité (CER). Exemple de code :  

```python
import fitz, pytesseract
doc = fitz.open("exemple.pdf")
for page in doc:
    text = page.get_text().strip()
    if not text:
        # PDF scanné : lancer OCR sur l’image
        img = page.get_pixmap(dpi=300)
        text = pytesseract.image_to_string(img.tobytes(), lang='fra')
    print(text[:100])  # afficher début du texte extrait
```  

Ce prototype initial sert à valider les outils sélectionnés.  

**Étape 3 : détection de structure**. Intégrer ensuite la détection de tables et zones clés. Par exemple : exécuter Camelot sur les PDF test. Comparer avec Tabula et pdfplumber. Pour OCR+table, tester DocTR ou PaddleOCR-VL pour voir s’ils extraient directement le JSON attendu.  

**Étape 4 : extraction d’entités**. Appliquer un modèle NER (spaCy fr ou LayoutLM fine-tuné) sur le texte extrait. Par exemple, détecter les entités « Personne », « Date », « Total » dans un document financier. Evaluer F1 localement sur un petit sous-ensemble étiqueté manuellement.  

**Étape 5 : itération et évaluation**. Comparer quantitativement (precision/recall, temps d’exécution) entre les solutions (ex. PyMuPDF+Tesseract vs Azure). Ajuster les paramètres (ex. binarisation, configuration PDFMiner). Implémenter la boucle de validation (scores de confiance, révision manuelle) pour améliorer la fiabilité.  

**Étape 6 : industrialisation**. Une fois les choix validés, dockeriser le pipeline complet (OCR, table, NER). Mettre en place la mise à l’échelle (cluster GPU si nécessaire) et le système de monitoring.  

**Exemple de pseudocode simplifié du pipeline** :  
```
for pdf_file in liste_documents:
    pages = open_pdf(pdf_file)
    for page in pages:
        if page.has_text():
            text_blocks = parse_text_positions(page)
        else:
            image = render_page_image(page)
            text_blocks = ocr_model.recognize(image)
        layout = layout_model.detect(text_blocks)
        tables = []
        for region in layout.tables:
            table = extract_table(region)   # Camelot ou TableTransformer
            tables.append(table)
        entities = ner_model.extract(text_blocks)
        data = merge_results(text_blocks, tables, entities)
        save_to_json(data, output_path)
```

**Risques & mitigations** :  
- *Scanne mal oucr.* → risque de mauvaise extraction sur documents très bruités. **Mitigation** : tester plusieurs prétraitements (deskew, diff filters) et éventuellement envelopper par un autre OCR comme ABBYY ou Azure pour comparaison.  
- *Temps de traitement élevé* → pipeline DL lourd peut être lent. **Mitigation** : profilage du code, traitement par lots, mise en cache. Possibilité d’enlever des modèles redondants (ex. ne pas lancer un 2ᵉ OCR si PDF natif).  
- *Nécessité de confidentialité* → ne pas envoyer certains documents vers le cloud. **Mitigation** : prévoir un mode d’exécution locale pour ces fichiers (ex. basculer sur Tesseract/PyMuPDF uniquement).  

**Extrait de code de test (non exhaustif)** :  
```python
import camelot
tables = camelot.read_pdf("exemple_table.pdf", flavor='lattice')
print("Tables détectées:", len(tables))
tables[0].to_csv("sortie.csv")
```
Ce pseudocode montre comment appeler Camelot pour extraire une table en CSV. Dans un prototype réel, chaque étape ci-dessus serait rendue paramétrable et journalisée pour permettre des ajustements rapides.

**Notes** : Tous les outils et approches mentionnés sont documentés dans la littérature et la documentation officielle (voir citations). Ce guide ne couvre pas les cas extrêmes (ex. écriture cursive complexe) où des solutions personnalisées peuvent être nécessaires.

**Sources principales** : documentation et publications officielles (Apache PDFBox【28†L55-L60】, LayoutLM【16†L69-L77】, PaddleOCR【45†L377-L385】, PubTables-1M【51†L116-L124】, FUNSD【55†L63-L72】, blog OpenNews sur l’extraction de tables【10†L93-L100】【9†L300-L308】, etc.). Ces références fournissent les informations techniques citées cidessus.