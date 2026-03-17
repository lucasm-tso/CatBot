# Optimisation du parsing PDF avec PaddleOCR

## Résumé exécutif 
**Prérequis:** Nous supposerons des PDF mixtes (images/scans) en langue indéterminée. Utiliser un pipeline complet: extraction d’images via pdfplumber/PyMuPDF【23†L43-L50】, suivi de prétraitement (redimensionnement, binarisation, désinclinaison)【14†L73-L81】【14†L98-L104】. Sélectionner des modèles adaptés (p.ex. détecteur DB, reconnaisseur PP-OCRv5)【14†L109-L116】【6†L254-L262】 et affiner par fine-tuning (~5000 exemples) pour le domaine【29†L259-L267】. Gérer la mise en page: multi-colonnes et tableaux via des pipelines de layout (PP-StructureV3, PaddleOCR-VL)【5†L798-L805】【26†L81-L90】. Appliquer un post-traitement: correction orthographique, dictionnaires métier et regex sur le texte brut. Évaluer l’OCR avec métriques de précision/recouvrement et benchmark modulaire【19†L58-L66】. Exploiter outils complémentaires: pdfplumber/PyMuPDF pour images, OCRmyPDF/Tesseract pour fallback【23†L43-L50】【8†L250-L258】. Pour la production, privilégier GPU, batchs efficaces, quantification (Slim) et export ONNX/OpenVINO【21†L67-L75】【19†L58-L66】. Les ressources clés incluent la doc officielle PaddleOCR, les articles techniques (ex. PaddleOCR 3.0【26†L81-L90】) et tutoriels.

## 1. Prétraitement des PDF 
- **Rasterisation:** Extraire chaque page en image haute-résolution (300–600 dpi), ou récupérer le texte vectoriel intégré si disponible. Bibliothèques recommandées: **pdfplumber** ou **PyMuPDF**【23†L43-L50】【8†L250-L258】.  
- **Redimensionnement:** Adapter la plus grande dimension à une taille fixe (640–960 px) tout en préservant l’aspect【14†L73-L81】. Laisser en couleur ou convertir en niveaux de gris selon besoin.  
- **Binarisation et débruitage:** Sur les scans, appliquer seuillage adaptatif (Otsu) et filtres (ex. médian, bilateral) pour améliorer contraste et lisibilité【16†L59-L67】【14†L98-L104】.  
- **Correction d’orientation:** Détecter/rectifier la rotation (0°/90°/180°/270°) automatiquement【6†L205-L213】【14†L98-L104】. PaddleOCR propose un module d’orientation (PP-LCNet_doc_ori à ~99% d’exactitude【6†L205-L213】).  
- **Alignement et découpage:** Détecter et corriger l’inclinaison (deskew). Segmenter les colonnes en régions distinctes (via heuristiques ou modèles de layout) pour faciliter la détection textuelle.

## 2. Modèles et configuration PaddleOCR 
- **Détecteur de texte:** Classiquement DB (Differentiable Binarization) pour la plupart des documents【14†L109-L116】. EAST/SAST sont alternatives pour certaines formes de texte.  
- **Reconnaisseur de texte:** Les modèles **PP-OCRv5** (mobile/serveur) supportent plusieurs langues et cas difficiles【6†L254-L262】. Le mode «_server_» favorise la précision (86% acc. moyenne) contre «_mobile_» pour la légèreté【6†L254-L262】.  
- **Layout/Parsing:** Pour documents complexes (multi-colonnes, tableaux, formulaires), utiliser **PP-StructureV3** (pipeline modulaire) ou le **PaddleOCR-VL** (modèle VL 0.9B ultra-compact)【5†L798-L805】【26†L81-L90】. Ces outils extraient blocs de texte, tableaux, formules, etc.  
- **Hyperparamètres clés:** Dans `PaddleOCR(... )`, régler `text_det_limit_side_len` (taille max), `box_thresh` (seuil confiance détection), `unclip_ratio` (factorisation de zone)【23†L125-L133】【33†L619-L627】. Par exemple, `box_thresh≈0.5–0.6` et `unclip_ratio≈1.5` sont courants【33†L619-L627】.  
- **Fine-tuning:** Si domaine ou alphabet particulier (chiffres vs lettres, langues rares), fine-tuner le reconnaisseur sur ~5k images annotées【29†L259-L267】. On peut restreindre le vocabulaire cible pour réduire les confusions (0 vs O, 1 vs I)【29†L259-L267】.  
- **Prédétection de langues/multi-langues:** Charger le paramètre `lang` approprié (`ch`, `en`, `multilang`…). Pour le français/d'autres langues, les modèles multilingues intégrés conviennent. 

## 3. Pipeline multi-colonnes, tableaux et formulaires 
PaddleOCR permet de chaîner les modules. Exemple de flux (mermaid ci-dessous) :
```mermaid
graph LR
  A[PDF scanné/mixte] --> B[Rasterisation (pdfplumber/PyMuPDF)]
  B --> C{Prétraitement}
  C --> D(Orientation et deskew)
  C --> E(Binarisation, débruitage)
  D --> F[Détection de texte (DB/EAST)]
  E --> F
  F --> G[Reconnaissance (PP-OCRv5)]
  G --> H[Post-traitement (corrections, regex)]
  H --> I[Sortie structurée (JSON/Markdown)]
```
Pour les **documents multi-colonnes**, extraire chaque colonne comme région distincte avant OCR, ou utiliser la détection de blocs de PaddleOCR. Pour les **tableaux**, activer `pp_structurev3_table` dans PP-StructureV3. Les **formulaires/notes** peuvent nécessiter une détection de champs (regex, PP-ChatOCR/KIE). 

## 4. Post-traitement OCR 
- **Correction d’erreurs:** Appliquer des règles métier et correcteurs orthographiques (dictionnaire, modèle de langue) pour corriger les fautes fréquentes. Ex.: remplacer “2O00” par “2000”, ou corriger les accents/français.  
- **Regex et heuristiques:** Utiliser des expressions régulières pour extraire champs structurés (dates, montants, N° de facture, etc.) et valider le format.  
- **Filtres lexicaux:** Restreindre le jeu de caractères (p. ex. chiffres seulement dans des champs numériques) pour forcer la reconnaissance correcte.  
- **Langage contextuel:** Intégrer un modèle de langue (ex. ERNIE/PaddleOCR-VL) pour améliorer la cohérence des segments extraits en contexte.

## 5. Évaluation et tests 
- **Métriques OCR:** Mesurer l’Hmean sur détection et l’exactitude caractère/mot en reconnaissance (CER/WER). Évaluer sur un ensemble test représentatif de PDF (divers formats).  
- **Protocoles:** Prévoir des jeux de tests multi-pages, multi-colonnes et mixtes. Vérifier la qualité de la **reconstitution du texte et l’ordre de lecture**.  
- **Benchmarks:** PaddleOCR 3.0 fournit des résultats comparatifs sur OmniDocBench, etc.【26†L81-L90】. Le *fine-grained benchmark* de la v3.2 aide à repérer précisément les goulets (ex. 60% du temps en détection)【19†L58-L66】. 

## 6. Outils et bibliothèques complémentaires 
- **pdfplumber, PyMuPDF:** Extraits fiables d’images ou de textes depuis PDF (utiles si PaddleOCR rate du texte vectoriel)【23†L43-L50】【8†L250-L258】.  
- **OCRmyPDF:** Pipeline pour convertir un PDF scanné en PDF *searchable* (utilise Tesseract en coulisses). Peut précéder ou suivre PaddleOCR.  
- **Tesseract:** OCR alternatif open source, parfois combiné (pour comparaison de résultats ou PDF très simples).  
- **PyMuPDF/pdfrw/PyPDF2:** Manipulation des documents (fusion, découpage, correction de DPI).  
- **OpenCV:** Pour les transformations d’images (rotation, érosion/dilatation, filtres) dans le prétraitement custom. 

## 7. Conseils de performance et déploiement 
- **GPU vs CPU:** Les GPU accélèrent nettement la détection et reconnaissance. En CPU, privilégier les modèles *mobile-lite*.  
- **Batching:** Si possible, traiter par lots ou paralléliser les pages. PaddleOCR ne fait pas nativement le *batch* pour plusieurs images, mais on peut utiliser multiprocessing.  
- **Quantification:** Utiliser les outils *Slim* pour quantifier les modèles (INT8) sur edge/mobile【20†L5-L8】.  
- **Export ONNX/OpenVINO:** Exporter les modèles Paddle vers ONNX, puis vers OpenVINO pour accélérer sur CPU Intel【21†L67-L75】. OpenVINO permet de compiler et optimiser pour CPU/VPUs.  
- **Tensorrt/FP16:** Sur GPU NVIDIA, tester FP16/TensorRT pour accélérer l’inférence.  
- **Profilage:** Activer `PADDLE_PDX_PIPELINE_BENCHMARK` (v3.2+) pour mesurer le temps de chaque module【19†L58-L66】 et cibler les optimisations.

## 8. Exemples de commandes et snippets 

- **CLI PaddleOCR:**  
  ```bash
  paddleocr --image_dir doc.pdf --use_angle_cls False --use_space_char True
  ```  
  (Remplacer par `-i fichier.pdf`, ajouter `--use_table_detector`, etc.)  
- **Python PaddleOCR:**  
  ```python
  from paddleocr import PaddleOCR
  ocr = PaddleOCR(lang='fr', text_det_limit_side_len=960, text_det_box_thresh=0.5)
  result = ocr.ocr("page1.png", use_gpu=True)
  ```  
  【23†L125-L133】  
- **Pipeline PP-StructureV3 (YAML):** Dans `configs/`, ajuster fields comme `box_thresh:0.6`, `unclip_ratio:1.5`【33†L619-L627】, et `use_gpu: True`【33†L521-L529】. Exemple de config JSON :  
  ```yaml
  Global:
    use_gpu: True
    epoch_num: 100
  Architecture:
    algorithm: "DB"
    ...
  PostProcess:
    name: "DBPostProcess"
    box_thresh: 0.6
    unclip_ratio: 1.5
  ```  
  (Autres paramètres voir examples officiels).  

## 9. Études de cas / benchmarks comparatifs 
- **PaddleOCR-VL vs StructureV3:** Des tests sur des PDF scientifiques (multi-colonne, math) montrent que PP-StructureV3 *lightweight* est très rapide (~2 s/page) avec de bonnes sorties LaTeX, alors que les modèles VLM (PaddleOCR-VL) prennent plus de temps (min)【26†L81-L90】.  
- **OCR de factures ou reçus:** Cas d’usage typique démontrent la nécessité de fine-tuning et de post-traitement. Par exemple, des études sur numéros de plaque confirment l’argument que **binarisation (Otsu)** et filtrage (bilateral) améliorent l’OCR dans des conditions réelles【16†L59-L67】.  
- **Comparaison avec Tesseract:** PaddleOCR tend à mieux gérer les polices variées et les langues multiples, mais Tesseract + OCRmyPDF peut être suffisant pour des PDF très simples et s’intègre directement dans des workflows PDF.

## 10. Ressources principales 
- **Documentation officielle:** [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) et [docs.paddleocr.ai](https://www.paddleocr.ai) pour guides, listes de modèles et pipeline【5†L798-L805】【6†L205-L213】.  
- **Articles académiques:** Rapport PaddleOCR 3.0 (arXiv) et PaddleOCR-VL【26†L81-L90】.  
- **Tutoriels et blogs:** «From Pixels to Words: PaddleOCR Pipeline» (explique prétraitement et détecteurs)【14†L73-L81】【14†L98-L104】, blog PaddleOCR v3.2 sur le benchmarking【19†L58-L66】, et tutoriels PaddleX pour modules prédéfinis.  
- **Forums/Q&A:** Issues GitHub (support technique et astuces) et tutos communautaires pour des cas d’usage spécifiques.  

**Tableau comparatif (exemples)**:

| Composant         | Options                | Points clés                                                |
|-------------------|------------------------|------------------------------------------------------------|
| Détection         | PP-OCRv5 (DB), EAST    | DB: robustesse/forme libre; EAST: plus rapide sur segments rectangulaires |
| Reconnaissance    | PP-OCRv5 (server/mobile)  | Serveur: haute acc. multi-langues; Mobile: léger/rapide    |
| Layout            | PP-StructureV3, PaddleOCR-VL | V3: pipeline modulaire (tables, formules); VL: modèle VLM tout-en-un, polyglotte |
| Prétraitement     | Otsu, CLAHE, BFilter   | Otsu: bon pour textes sur fond clair; CLAHE: améliore contraste【16†L59-L67】   |
| OCR complémentaire| Tesseract/OCRmyPDF     | Document search – faible à moy. qualité, insertion texte PDF. |

**Diagramme pipeline:** (voir schéma Mermaid ci-dessus). Les timelines de déploiement dépendront du projet; typiquement prévoir 2–4 semaines pour tests, fine-tuning et intégration.

**Sources citées:** Documentation PaddleOCR officielle【5†L798-L805】【6†L205-L213】, tutoriels techniques【14†L73-L81】【14†L98-L104】【19†L58-L66】【23†L43-L50】, articles de recherche【16†L59-L67】【26†L81-L90】 et échanges GitHub【29†L259-L267】【33†L619-L627】. Ces références couvrent les pratiques recommandées, les choix de modèles, ainsi que les outils associés.