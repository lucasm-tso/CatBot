# Résumé exécutif  
Les modèles vision-langage récents permettent de traiter des images et du texte de façon conjointe, mais la plupart fonctionnent selon une inférence **statique** (l’image entière est encodée d’un coup), limitant les capacités d’exploration ou de recherche itérative. Plusieurs approches émergent toutefois pour introduire un raisonnement en plusieurs étapes sur l’image : boucles de question-réponse successives, attention récursive, utilisation d’outils (OCR, segmentation) ou mécanismes de retour d’information (verifier/rationale). Des modèles tels que Flamingo ou GPT-4o (OpenAI) intègrent déjà des architectures cross-attention permettant un raisonnement plus fin【57†L248-L256】【68†L120-L124】. Des travaux récents (par ex. Bai *et al.*, 2025) proposent de modéliser la recherche d’information visuelle comme un processus itératif (MDP) avec un “reasoner” qui propose des actions sur l’image et un “verifier” entraîné (DPO) pour décider de poursuivre【61†L236-L244】. Des pipelines basés sur la détection d’objets (Detectron2, MMDetection) combinés à des LLM, ou des approches RAG spécifiquement adaptées aux données visuelles (VisRAG), sont également explorés【70†L69-L74】【75†L317-L321】. La faisabilité de telles recherches multi-étapes dépend des modèles utilisés (CLIP, BLIP-2, Kosmos, LLaVA, PaLIGemma, etc.), des benchmarks disponibles (VQA, GQA, TextCaps, RefCOCO, Visual7W, OK-VQA, etc.), et des limitations techniques (coût de calcul, hallucinations, manque de supervision multi-étapes). Ce rapport analyse en détail ces dimensions, présente des pipelines d’implémentation, propose des expériences d’évaluation et recense les ressources open-source pertinentes pour mettre en œuvre et tester une recherche récursive sur image avec des modèles vision-langage.

## Contexte technique et définitions  
Une *recherche récursive sur une image* désigne ici une procédure où l’IA interagit plusieurs fois avec l’image (ou le modèle) pour affiner sa réponse, plutôt qu’une seule passe statique. On parle aussi d’**affinement itératif**, de *question-réponse multi-étapes* ou de *vision active*. Par exemple, au lieu de demander « Qui est sur la photo ? » et d’obtenir directement la réponse, le système peut poser plusieurs questions successives ou se focaliser sur des sous-régions de l’image. L’**attention multi-étape** ou *chain-of-thought* en vision consiste à décomposer le problème visuel en étapes, à l’image de ce qui se fait en traitement du langage【63†L65-L74】. La *recherche basée sur région* (region-based search) implique de segmenter l’image (par ex. via détection d’objets ou segmentation sémantique) et d’extraire des informations (légendes, textes OCR) avant chaque nouvelle requête. Le concept de **vision active** recouvre des scénarios où l’agent peut modifier sa vue (changer de point de vue, zoomer), typique en robotique ou drones【73†L128-L136】.  

Plusieurs familles de modèles sont concernées : les modèles *CLIP*-like fournissent des embeddings image/texte pour la recherche d’images par similarité【14†L65-L74】. Les architectures *multimodales à un seul flux* (single-tower, e.g. Perceiver, Flamingo) ou *double flux* (ex. CLIP-LLM) combinent encodeurs visuels (CNN, ViT) et LLMs. Par exemple, Flamingo (DeepMind) utilise un LLM gelé avec des couches de cross-attention pour fusionner image et texte【57†L248-L256】. D’autres tels BLIP-2 (Salesforce) ou LLaVA (HarveyLabs) ajoutent un module intermédiaire (Q-Former ou simple projection) pour encoder l’image avant l’LLM【23†L74-L82】. OpenAI GPT-4o (“omni”) accepte en entrée toute combinaison de texte, image et audio en temps réel【68†L120-L124】, fournissant un cadre très large pour le raisonnement multimodal. Enfin, des modèles spécialisés comme Kosmos-2/2.5 (Microsoft) se concentrent sur la reconnaissance de texte dans l’image (OCR et structure)【25†L64-L72】【80†L254-L263】, offrant des mécanismes de *grounding* texte-image.  

Les concepts de *vérification* et *feedback* sont clés pour la récursivité. Par exemple, le framework VTS-V (NeurIPS 2025) formalise l’inférence itérative comme un Processus de Décision Markovien, où un **reasoner** propose des actions visuelles (par ex. explorer une région, zoomer) et un **verifier** (entraîné par DPO) évalue si l’itération doit se poursuivre【61†L236-L244】. Dans ce cadre, chaque étape produit un résumé partiel (caption, OCR, etc.) envoyé à un LLM pour guider l’étape suivante. Les approches *retrieval-augmented* (RAG) multimodales, comme VisRAG【70†L69-L74】, utilisent un VLM pour indexer l’image ou ses vues et interroger cette base pour compléter la génération de réponses. Enfin, la technique de **chain-of-thought** (CoT) provennant du langage peut être adaptée en visuel, soit en explicitant des étapes intermédiaires sous forme de texte, soit en utilisant des modules prédictifs (par ex. prédiction de la prochaine image en CoT-VLA pour la robotique【52†L72-L80】).  

## Méthodes existantes et travaux récents  
**Approches par modèles vision-langage**  
- **CLIP** (OpenAI, 2021) a popularisé la recherche d’images par similarité texte-image via des embeddings contrastifs, mais opère sur l’image entière. Des travaux ont montré que découper l’image en sous-régions (grilles 3×3, etc.) avant d’effectuer la recherche peut améliorer la précision de récupération pour certaines tâches spécialisées【14†L65-L74】.  
- **Flamingo** (DeepMind, 2022) utilise un grand LLM gelé (à l’époque Gopher), épaulé par un encodeur visuel finetuné. Il introduit des couches de cross-attention pour fusionner image et texte en profondeur【57†L248-L256】. Flamingo est performant sur des tâches variées (légendes d’images, VQA, etc.) avec peu d’exemples grâce à cette architecture hybride.  
- **Modèles instruction-tunés** tels que **LLaVA** (Large Language and Vision Assistant, Harvey Labs, 2023) ou **PaliGemma** (Google, 2025) affinent les modèles visuels avec de grandes quantités de paires image-question/réponse en style conversationnel. LLaVA par exemple atteint des performances de pointe sur 11 benchmarks (y compris GQA, OK-VQA) en fusionnant un LLaMA avec un encodeur d’image【23†L74-L82】. Paligemma 2 mix, annoncé par Google en 2025, est un VLM multi-tâches (OCR, segmentation, captioning, VQA) disponible via Hugging Face/Keras【27†L135-L139】【27†L150-L156】.  
- **GPT-4o (OpenAI, 2024)** est un modèle « omni » temps réel, capable de traiter texte, audio, image et vidéo. Dans la page officielle OpenAI, il est décrit comme un système unifié (même réseau neuronal) surpassant GPT-4 en compréhension visionnelle et audio, avec une latence proche du temps humain【68†L120-L124】【68†L139-L148】. GPT-4o peut être utilisé en mode conversationnel multi-tours avec images, ce qui se prête à des recherches itératives (via l’API ChatGPT Vision par exemple).  
- **Kosmos-1/2 (Microsoft)** vise le *grounding* multi-modal. Flamingo a inspiré Kosmos-1, un modèle instruction-tuné multi-modal axé sur des tâches comme OCR, captioning, VQA【57†L259-L265】. Kosmos-2/2.5 se focalise sur l’analyse de documents et d’images riches en texte : il génère des blocs de texte avec coordonnées spatiales ou formatés en markdown à partir d’images textuelles【25†L64-L72】【80†L254-L263】. Ces modèles apportent des mécanismes de lien entre texte et régions visuelles (visual grounding).  

**Architectures et mécanismes pour le raisonnement itératif**  
- **Chaînes d’attention récursives** : certains réseaux récursifs apprennent à focaliser successivement sur différentes parties de l’image. Par exemple, le *Recurrent Visual Attention Model* (RAM) historique utilisait une politique RL pour sélectionner un patch d’image à chaque pas. Des travaux plus récents comme MRAN-VQA (2025) développent des mécanismes d’attention récursive pour le VQA, qui affinent de façon hiérarchique les représentations en plusieurs étapes【7†L416-L424】.  
- **Chain-of-Thought en visuel** : Inspiré des LLM, on force le modèle à expliciter des étapes de raisonnement. CoT-VLA (2025) applique ce principe pour la robotique, en demandant au modèle de prédire des images-cibles intermédiaires (« visual chain of thought ») avant d’effectuer des actions【52†L72-L80】. En traitement d’images statiques, on peut solliciter un LLM avec des prompts détaillant chaque sous-question (ex. « décris d’abord ce que tu vois, puis cherche l’objet », etc.), ce qui améliore la fiabilité du raisonnement【63†L65-L74】.  
- **Prompting itératif** : L’LLM peut être sollicité plusieurs fois en boucle. Par exemple, on pose une première question au modèle, on observe sa réponse (éventuellement relative à une sous-région détectée), puis on lui demande de reformuler la question ou d’approfondir. Cette technique, couplée à des prompts de style CoT, permet de simuler un raisonnement multi-tours. Des travaux récents (comme VTS-V) vont plus loin en formalisant cela comme un MDP【61†L236-L244】 : un *reasoner* propose une action (p. ex. zoom, recadrage, etc.), un *verifier* juge si le raisonnement peut s’arrêter, et on forme les deux via apprentissage par préférence (DPO).  
- **Boucles d’outils visuels** : Il est courant de combiner des outils spécialisés pour enrichir l’étape de recherche. Par exemple, **Detectron2**【32†L312-L320】 ou **MMDetection**【34†L369-L372】 peuvent détecter et segmenter les objets dans l’image. On peut alors appliquer un modèle de légende d’image (captioning) ou d’OCR sur chaque région détectée pour extraire du texte ou des labels, avant de passer ces informations au LLM. L’algorithme boucle ainsi : détection -> description/caption -> question/réponse. Le projet LLaVA-Plus (2024) illustre cette idée en apprenant à utiliser des « skills » (outils) dans un assistant VLM.  
- **Retrieval-Augmented (RAG)** : Au lieu de ne tirer que du contenu textuel, RAG peut être étendu aux données visuelles. VisRAG (2024) propose un pipeline où un document complexe est encodé par un VLM comme une image entière puis indexé pour la recherche【70†L69-L74】. De même, on peut indexer une base d’images par CLIP ou utiliser des vecteurs visuels pour récupérer des images similaires ou des passages de textes associés, améliorant les réponses du LLM. Cette approche maximise l’utilisation des informations visuelles (mise en page, images) au lieu d’un simple prétraitement textuel【70†L69-L74】.

**Jeux de données et benchmarks**  
Plusieurs corpus alimentent l’étude du raisonnement visuel multi-étapes : 
- **VQA** (Visual Question Answering, Antol *et al.*, 2015) propose ~0.25M images et ~0.75M questions libres sur les images, couvrant dénombrement, attributs, objets【38†L64-L72】.  
- **GQA** (Hudson & Manning, 2019) est un jeu plus orienté raisonnement compositional, avec 22M de questions générées à partir de graphes de scène, et de nouveaux métriques (consistance, ancrage, plausibilité)【40†L65-L73】.  
- **Visual7W** (Zhu *et al.*, 2016) étend VQA en associant à chaque question/réponse la zone (bounding box) dans l’image (« grounding »). Le jeu contient 327 939 QA sur 47 300 images, avec 1,311M de choix multiples et 561 459 annotations de localisation【45†L93-L100】.  
- **TextCaps** (Sidorov *et al.*, ECCV 2020) cible la description d’images contenant du texte, avec 145k légendes pour 28k images. Le modèle doit lire les textes visibles et décider quoi en retranscrire dans la légende【50†L69-L73】.  
- **RefCOCO/+/g** sont des bases d’*expressions référentielles* où chaque phrase (ex. « la fille portant un sac à dos ») correspond exactement à une région unique dans l’image【43†L1232-L1240】. Elles sont destinées à tester la capacité de localisation à partir d’une description.  
- **OK-VQA** (Marino *et al.*, CVPR 2019) pose des questions nécessitant des connaissances externes (ex. culture générale) en plus de l’image. Le dataset contient ~14 000 questions, soulignant la difficulté pour un VLM pur de répondre sans information extérieure【47†L69-L73】.  
- Au-delà, des benchmarks pour vision active ou multi-tours émergent : *FlySearch* (Pardyl *et al.*, 2025) simule un drone devant trouver des objets (voitures, incendies, personnes disparues, etc.) dans de vastes scènes 3D, et révèle l’écart de performance entre VLMs et humains en exploration【73†L128-L136】. Le rapport VTS (Bai *et al.*, NeurIPS 2025) fournit aussi des trajectoires supervisées et des comparaisons préférentielles pour l’inférence multi-étapes【61†L236-L244】.  

## Pipelines d’implémentation et pseudo-code  

Pour mettre en œuvre une recherche récursive, on construit typiquement un pipeline combinant encodeur visuel, outils spécialisés et LLM. En voici un exemple générique :

```mermaid
flowchart LR
    IMG[Image d'entrée] --> VM(Modèle Vision-Langage)
    VM --> ROI(Proposition de régions d'intérêt)
    ROI --> TOOL(Outils visuels (OCR, segmentation))
    TOOL --> LLM(Modèle de raisonnement (LLM))
    LLM --> Query(Requête ou réponse affinée)
    Query --> ROI
```

Un pseudo-code Python illustrant ce principe : 

```python
# Pseudo-code : Recherche visuelle itérative avec VLM
image = load_image("photo.jpg")
query = "What is the man doing?"
max_steps = 5

for step in range(max_steps):
    # 1) Extraire des régions pertinentes selon la requête actuelle
    regions = vision_model.detect_regions(image, context=query)
    # 2) Appliquer des outils sur chaque région (caption, OCR, etc.)
    infos = []
    for region in regions:
        caption = vision_model.caption(region)
        ocr_text = vision_model.read_text(region)
        infos.append((caption, ocr_text))
    # 3) Raisonner avec l'LLM pour obtenir réponse et nouvelle requête
    answer, query = language_model.reason(query, infos)
    if is_final_answer(answer, query):
        break

print("Réponse finale :", answer)
```

Ici, `vision_model` peut être un modèle comme CLIP ou un ViT donnant accès à la détection d’objets et à la génération de légendes/OCR sur une sous-image. Le `language_model` est le LLM (GPT-4o Vision, LLaVA, etc.) qui prend en entrée la requête initiale et les informations extraites pour formuler une réponse et/ou raffiner la question. Par exemple, on pourrait utiliser le pipeline BLIP-2 de Hugging Face pour la détection et le captioning, puis GPT-4o ou LLaVA pour le raisonnement langagier, en loop.  

En pratique, plusieurs pipelines concrets existent :  
- **Pipeline CLIP+LLM** : on encode l’image en vecteurs CLIP pour toutes les sous-régions (ex. grille 3×3), on mesure la similarité avec la requête textuelle pour sélectionner la ou les régions les plus pertinentes, puis on les décrit avec un captioner ou en pose VQA au LLM. Ce processus peut être répété en affinant la requête.  
- **Pipeline Detecteur + Captioner + LLM** : on utilise Detectron2 ou MMDetection pour localiser des objets (personnes, voitures, etc.). Chaque bounding box est passée à un module de captioning (BLIP-2, GiT, etc.) ou à OCR (pour lire du texte)【32†L312-L320】【34†L369-L372】. Les descriptions produites sont ensuite fournies à un LLM (via conversation ou prompt) qui les intègre pour répondre à la question globale.  
- **Pipeline RAG multimodal** : on encode l’image ou un document visuel avec un VLM (p. ex. en extrayant plusieurs patchs ou en traitant l’image comme “document”), on utilise ensuite un moteur de recherche vectoriel pour récupérer des contenus similaires ou pertinents (images annotées, textes associés), puis on injecte ces résultats dans le LLM. VisRAG en est un exemple : le document est indexé comme image et les étapes de recherche alimentent la génération【70†L69-L74】.  
- **LangChain/Agents multimodaux** : des frameworks comme LangChain proposent de combiner prompts, APIs et fonctions visuelles. Par exemple, un agent peut demander à GPT-4o de “regarder l’image”, puis d’appeler localement un segmentateur (SAM) ou un OCR selon les instructions, en boucle.  

Chaque pipeline dépend du contexte. Par exemple, un usage simple pour extraire des informations d’un GUI pourrait être réalisé avec GPT-4o Vision seul (API OpenAI) en posant une série de prompts (question initiale, puis « donne-moi plus de détails… »). En revanche, des scénarios exigeant du contrôle fin (robotique, pipelines fermés) utilisent des modèles open-source (LLaVA-Plus, Kosmos-2.5, etc.) associés à des bibliothèques d’IA classique (Detectron2, SAM, etc.).

## Plan d’expérimentation et évaluation  

Pour évaluer formellement la capacité de *recherche récursive*, plusieurs protocoles et métriques peuvent être conçus :  

- **Exactitude finale** : pour les tâches VQA existantes (VQA, GQA, Visual7W, OK-VQA), on mesure si la réponse finale est correcte (mesure « accuracy » usuelle). On peut comparer la version « statique » (question unique) avec l’approche itérative (multi-tours).  
- **Métriques de cohérence et d’ancrage** : sur des bases comme GQA, on peut utiliser les scores proposés (consistency, grounding, plausibility)【40†L65-L73】. Par exemple, on vérifie si les sous-réponses intermédiaires restent consistantes entre elles et si les objets mentionnés sont bien ancrés aux bonnes régions.  
- **Nombre d’étapes et efficacité** : on suit le nombre moyen d’itérations nécessaires pour converger vers la réponse. Un système efficace nécessitera moins de tours pour atteindre la même exactitude. On peut fixer un budget d’étapes maximal (ex. 5) et évaluer la couverture de la réponse.  
- **Temps de latence** : les pipelines itératifs sont plus lents qu’une inférence unique. Il faut mesurer le temps moyen de réponse (pour une question type) et évaluer la viabilité pratique.  
- **Robustesse et ablations** : on peut tester la résilience du processus itératif aux erreurs intermédiaires (ex. hallucination du LLM à une étape). On compare par exemple une chaîne CoT normale vs une chaîne CoT avec un vérificateur supplémentaire (DPO). On peut aussi ablater des composants (supprimer le module OCR ou le verifier) pour mesurer leur impact.  
- **Tests multi-tours spécifiques** : créer des scénarios nécessitant explicitement plusieurs pas, par exemple des questions où la réponse se construit par recoupement d’indices dispersés dans l’image ou du texte partiel. Ces tests peuvent être synthétiques ou dérivés des dataset existants (ex. dériver des sous-questions de GQA).  
- **Utilisation des datasets dédiés** : le jeu VTS introduit des trajectoires supervisées pour l’entraînement (VTS-SFT) et des comparaisons par préférence (VTS-DPO)【61†L236-L244】. On peut s’en servir pour entraîner et comparer différents algorithmes d’inférence dynamique.  
- **Évaluation qualitative** : en plus des métriques quantifiées, des évaluations humaines peuvent juger la clarté du raisonnement pas-à-pas et la sensibilité aux instructions.  

En résumé, on combinera des tâches standards de QA visuel avec de nouvelles mesures orientées multi-étapes. L’hypothèse est que l’approche itérative doit améliorer l’exactitude sur les questions complexes (p. ex. GQA, OK-VQA) au prix d’une latence supplémentaire. Des ablations permettront de quantifier l’apport de chaque mécanisme (CoT, outils visuels, feedback).

## Référentiels open-source et APIs  

| **Outil / API**              | **Type**                | **Avantages**                                                     | **Limites**                                                        |
|------------------------------|-------------------------|-------------------------------------------------------------------|--------------------------------------------------------------------|
| [GPT-4o Vision (OpenAI)](https://openai.com/index/hello-gpt-4o/) | API vision-langage      | Solution clé-en-main, multi-modale (texte, image, audio), très puissant【68†L120-L124】, réponse en temps réel | Fermé/propriétaire, coût payant, dépendance au cloud, confidentialité |
| [Hugging Face Transformers](https://huggingface.co/models) | Bibliothèque de modèles VLM  | Grand choix de modèles (BLIP-2, LLaVA, PaLIGemma, Kosmos, etc.)【23†L74-L82】【27†L150-L156】, open-source, fine-tuning possible | Utilise des modèles lourds (besoin de GPU), performances variables selon modèle |
| [Detectron2](https://github.com/facebookresearch/detectron2)【32†L312-L320】       | Détection/segmentation  | Bibliothèque SOTA de détection/segmentation (Facebook)【32†L312-L320】, bien documentée, nombreuses architectures | Configuration complexe, nécessite GPU puissant, conçu pour vision seule |
| [MMDetection](https://github.com/open-mmlab/mmdetection)【34†L369-L372】     | Détection/segmentation  | Outil OpenMMLab très complet, nombreux algorithmes (YOLO, Mask R-CNN…)【34†L369-L372】 | Lourde, peut être complexe à intégrer, overhead mémoire/temps |
| [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)【75†L317-L321】 | Segmentation universelle | Modèle promptable par points/boîtes pour segmentation d’objets【75†L317-L321】, zéro-shot efficace, code open-source | Ne fait que de la segmentation (pas de raisonnement), nécessite invites précises, lourd |
| [LLaVA (GitHub)](https://github.com/haotian-liu/LLaVA)【78†L348-L356】          | VLM instruction-tuning  | Code open-source pour assistant visuel, atteint SoTA (surpasse Gemini Pro)【78†L348-L356】, facile à entraîner sur données privées | Nécessite LLaMA ou Qwen, GPU costaud, dataloader complexe |
| [PaliGemma (Hugging Face)](https://huggingface.co/models?filter=paligemma)【27†L135-L139】 | VLM Google multi-tâches   | Modèle pré-entraîné multi-domaines (OCR, VQA, segmentation) prêt à l’emploi【27†L150-L156】 | Pas totalement open-source (API ou HF), très gros modèles, documentation limitée |
| [その他フレームワーク **Lightning, MindSpore…**]  | –                       | D’autres frameworks peuvent être utilisés pour la formation ou l’inférence de ces modèles| Variabilité de support et de maturité selon la librairie |

Chaque ligne du tableau ci-dessus renvoie vers le code source ou la documentation correspondante. Par exemple, Detectron2【32†L312-L320】 et MMDetection【34†L369-L372】 sont mentionnés dans leurs *README*, et SAM est officialisé par Meta avec des benchmarks zéro-shot impressionnants【75†L317-L321】. Ces outils peuvent être combinés : par exemple, utiliser Detectron2 pour générer des boîtes, puis SAM pour segmenter ou un captioner BLIP-2 sur ces régions, avant de passer le tout à un LLM comme GPT-4o ou LLaVA.  

## Limitations, risques et perspectives futures  

Les approches récursives sur images posent plusieurs défis :  

- **Coût de calcul et latence** : chaîner plusieurs étapes (détection, caption, LLM) multiplie le temps d’inférence et la demande GPU. Par exemple, GPT-4o ou LLaVA-1.5 nécessitent d’importantes ressources (GPU haut de gamme, mémoire) pour chaque tour de boucle. Cela limite l’applicabilité en temps réel.  
- **Hallucinations et incohérences** : les LLM peuvent produire des réponses erronées ou hors-sujet. En boucle, une erreur à une étape peut se propager. Des mécanismes de vérification (RLHF/DPO, comme dans VTS【61†L236-L244】) ou de fact-checking visuel sont nécessaires pour contrôler les dérapages. Les modèles actuels tendent à *inventer* des détails s’ils ne sont pas explicitement ancrés dans l’image.  
- **Raisonnement ancré** : relier précisément chaque morceau de la chaîne de raisonnement aux pixels d’origine est difficile. Les métriques d’ancrage (grounding) sont non triviales à optimiser. Par exemple, un LLM pourrait répondre correctement à une question sans identifier correctement l’objet dans l’image, ce qui reste un problème d’évaluation.  
- **Manque de données supervisées multi-étapes** : la plupart des datasets comme VQA ne fournissent qu’une question et une réponse finale. On manque de grands corpus annotés avec des raisonnement visuels explicites multi-tours (VTS est un pas dans cette direction). Ce déficit rend l’entraînement spécifique (par SFT ou RL) plus difficile.  
- **Biais et couverture** : les modèles pré-entraînés peuvent manquer de connaissances spécifiques (domaine médical, techniques, etc.) et reproduire des biais socioculturels. En ajoutant l’aspect multi-étapes, on introduit aussi des biais de prompt ou de stratégie (ex. tendance à répondre trop tôt, ne pas itérer quand il faudrait).  
- **Défauts d’interface** : combiner outils visuels et LLM introduit des points d’échec (p. ex. OCR qui rate des textes, segmenter mal un objet). L’intégration et la robustesse de ces chaines hétérogènes restent limitées.  

Pour l’avenir, plusieurs pistes sont ouvertes :  
- Développer des **modèles end-to-end dynamiques** qui intègrent nativement l’idée de multi-turn, plutôt que d’enchaîner des modules séparés. Par exemple, co-entraîner vision et langage avec un mécanisme de mémoire adaptative ou de croyance (comme VTS-V) pour simuler l’« attention permanente » sur l’image.  
- Enrichir les benchmarks et les métriques pour mieux prendre en compte les aspects interactifs. Par exemple, mesurer la cohérence multi-étapes, la pertinence de chaque requête soumise, ou introduire des tâches plus complexes (fusionner plusieurs images, questionnaires séquentiels, etc.).  
- Intégrer plus de **connaissances externes** : associer vision et bases de connaissances (images+texte+Wiki) pour que le modèle accède à des faits lors de la recherche. Par exemple, coupler un pipeline RAG multimodal sur des documents liés à l’image.  
- Réduire la latence via des modèles légers quantifiés (4-8 bits) ou des architectures hiérarchiques où seuls les résolutions/patchs pertinents sont analysés en profondeur.  
- Étudier la **fiabilité et la sécurité** : vérifier formellement que le processus itératif ne génère pas de réponses inadéquates (p.ex. dans des contextes sensibles comme la médecine ou la conduite autonome).  

En résumé, si la recherche itérative sur images avec des VLMs est prometteuse (capable de réponses plus fines et de justification explicite), elle reste encore dans un stade exploratoire. Les progrès récents (CoT visuel, benchmarks comme VTS, modèles multi-modaux puissants) ouvrent la voie, mais la robustesse, la scalabilité et l’explicabilité des solutions doivent être renforcées pour une adoption pratique.  

**Sources :** Nous nous sommes appuyés sur la littérature récente et la documentation officielle (articles de conférence NeurIPS/CVPR 2025, preprints arXiv, blogs Google/OpenAI, repos GitHub) pour cette synthèse【61†L236-L244】【57†L248-L256】【68†L120-L124】【27†L150-L156】【75†L317-L321】【70†L69-L74】. Chaque affirmation technique est citée par la référence correspondante pour garantir la rigueur scientifique.