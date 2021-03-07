<h1 align="center">
<p><a href="https://www.kaggle.com/c/nlp-getting-started">Real or Not? NLP with Disaster Tweets</a></p>
</h1>

## Kaggle Competition Description:
This section is taken verbatim from the [kaggle competition page](https://www.kaggle.com/c/nlp-getting-started)

>Twitter has become an important communication channel in times of emergency.
>The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).
>
>But, it’s not always clear whether a person’s words are actually announcing a disaster. Take this example:

![](reports/figures/ablaze.png)


>The author explicitly uses the word “ABLAZE” but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine.
>In this competition, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. You’ll have access to a dataset of 10,000 tweets that were hand classified. If this is your first time working on an NLP problem, we've created a [quick tutorial](https://www.kaggle.com/philculliton/nlp-getting-started-tutorial") to get you up and running.
>Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.</p></div>

> ### Acknowledgments
>This dataset was created by the company figure-eight and originally shared on their ‘Data For Everyone’ website here.

>Tweet source: https://twitter.com/AnyOtherAnnaK/status/629195955506708480
## Project Organization
<details>
<summary><b>Hierarchy</b>
</summary>
<p>

```

.
|-- [       4096]  data
|   |-- [       4096]  external
|   |   |-- [       4096]  appen
|   |   |   |-- [     715427]  disaster_response_messages_test.csv
|   |   |   |-- [    5746561]  disaster_response_messages_training.csv
|   |   |   `-- [     739819]  disaster_response_messages_validation.csv
|   |   |-- [       4096]  figureeight
|   |   |   |-- [        181]  README.md
|   |   |   `-- [    2208398]  socialmedia-disaster-tweets-DFE.csv
|   |   |-- [       4096]  kaggle
|   |   |   |-- [      68603]  publicleaderboard.csv
|   |   |   |-- [      22746]  sample_submission.csv
|   |   |   |-- [     420783]  test.csv
|   |   |   `-- [     987712]  train.csv
|   |   |-- [       4096]  sentiment140
|   |   |   |-- [      74326]  testdata.manual.2009.06.14.csv
|   |   |   `-- [   85088272]  training.1600000.processed.noemoticon.zip
|   |   |-- [       4096]  slang
|   |   |   `-- [        280]  acronyms.json
|   |   `-- [       4096]  wikipedia
|   |       `-- [       3473]  emoticons.json
|   |-- [       4096]  features
|   |   |-- [       4096]  meta_embeddings
|   |   |-- [     365552]  test_6_topics_15_iterations.npy
|   |   |-- [    6682752]  test_ae_embeddings.npy
|   |   |-- [   13365376]  test_bert_large_cased_whole_embeddings.npy
|   |   |-- [   10024064]  test_bert_sst2_embeddings.npy
|   |   |-- [   10024064]  test_bertweet_embeddings.npy
|   |   |-- [    1670784]  test_nnlm_en_128_norm_embeddings.npy
|   |   |-- [    6682752]  test_use4_embeddings.npy
|   |   |-- [     365552]  train_6_topics_15_iterations.npy
|   |   |-- [     365552]  train_6_topics_3_iterations.npy
|   |   |-- [   15591552]  train_ae_embeddings.npy
|   |   |-- [   31182976]  train_bert_large_cased_whole_embeddings.npy
|   |   |-- [   23387264]  train_bert_sst2_embeddings.npy
|   |   |-- [   23387264]  train_bertweet_embeddings.npy
|   |   |-- [    3897984]  train_nnlm_en_128_norm_embeddings.npy
|   |   |-- [   15591552]  train_use4_embeddings.npy
|   |   `-- [       4096]  tweet_sentiment
|   |       |-- [      13180]  test_sent_lr_nought_1.npy
|   |       |-- [      13180]  test_sent_lr_nought_3_400_round.npy
|   |       |-- [      13180]  test_sent_lr_nought_3.npy
|   |       |-- [      30580]  train_sent_lr_nought_1.npy
|   |       |-- [      30580]  train_sent_lr_nought_3_400_round.npy
|   |       `-- [      30580]  train_sent_lr_nought_3.npy
|   `-- [       4096]  processed
|       |-- [     442585]  bigrams_with_frequency.graphml
|       |-- [     306981]  bigrams_with_frequency_largest_component.graphml
|       |-- [     265055]  cooccurence.gephi
|       |-- [    2353376]  cooccurrence.gexf
|       |-- [    1222571]  cooccurrence.json
|       |-- [      30730]  duplicates.gephi
|       |-- [      48744]  duplicates.graphml
|       |-- [       4096]  kaggle
|       |   `-- [    1024668]  train.csv
|       |-- [     198800]  network.pdf
|       |-- [    1686642]  processed.graphml
|       |-- [    1691020]  processed_normalized.graphml
|       `-- [     374524]  subgraph_stats.csv
|-- [        627]  docker-compose.yml
|-- [       4096]  dockerfiles
|   |-- [       4096]  gpu
|   |   `-- [        534]  Dockerfile
|   `-- [       4096]  vanilla
|       `-- [       1344]  Dockerfile
|-- [       4096]  docs
|   |-- [        489]  commands.rst
|   |-- [       7820]  conf.py
|   |-- [        256]  getting-started.rst
|   |-- [        443]  index.rst
|   |-- [       5114]  make.bat
|   `-- [       5600]  Makefile
|-- [      11357]  LICENSE
|-- [        951]  Makefile
|-- [       4096]  models
|   |-- [      22746]  bertweet_finetuned.csv
|   |-- [      22746]  bertweet_finetuned_v2.csv
|   |-- [       4096]  bertweet_finetuning
|   |   `-- [       4096]  bertweet_kaggle
|   |       |-- [        775]  config.json
|   |       `-- [  539876480]  tf_model.h5
|   |-- [    2900673]  model_2021-01-13_200919_Pipeline_1x10cv_0.75_bertweet.pck
|   |-- [    2898625]  model_2021-01-13_201242_Pipeline_1x10cv_0.77_use4.pck
|   |-- [    2895551]  model_2021-01-13_201437_Pipeline_1x10cv_0.75_nnlm_en_128_norm.pck
|   |-- [    2902721]  model_2021-01-13_201659_Pipeline_1x10cv_0.72_bert_large_cased_whole.pck
|   |-- [      22746]  submission_2021-01-13_200919_Pipeline_1x10cv_0.75_bertweet.csv
|   |-- [      22746]  submission_2021-01-13_201242_Pipeline_1x10cv_0.77_use4.csv
|   |-- [      22746]  submission_2021-01-13_201437_Pipeline_1x10cv_0.75_nnlm_en_128_norm.csv
|   |-- [      22746]  submission_2021-01-13_201659_Pipeline_1x10cv_0.72_bert_large_cased_whole.csv
|   |-- [      22746]  submission_2021-01-18_104600_Pipeline_1x10cv_0.78_bertweet.csv
|   |-- [      22746]  submission_2021-01-18_105116_Pipeline_1x10cv_0.78_bertweet.csv
|   |-- [      22746]  submission_2021-01-18_110410_Pipeline_1x10cv_0.78_bertweet.csv
|   |-- [      22746]  submission_2021-01-18_110619_Pipeline_1x10cv_0.78_use4.csv
|   |-- [      22746]  submission_2021-01-18_110839_Pipeline_1x10cv_0.78_use4.csv
|   |-- [      22746]  submission_2021-01-18_110850_Pipeline_1x10cv_0.78_use4.csv
|   |-- [      22746]  submission_2021-01-20_135736_Pipeline_1x10cv_0.76_bert_sst2.csv
|   |-- [      22746]  submission_2021-01-20_144757_Pipeline_1x10cv_0.74_bertweet.csv
|   |-- [      22746]  submission_2021-01-21_202752_Pipeline_1x10cv_0.77_ae.csv
|   |-- [      22746]  submission_2021-01-21_204226_Pipeline_1x10cv_0.76_ae.csv
|   |-- [      22746]  submission_2021-01-21_210502_Pipeline_1x10cv_0.76_ae.csv
|   `-- [       4096]  xgboost
|       |-- [     609581]  bst_lr_nought_1.dump
|       |-- [     582574]  bst_lr_nought_1.save
|       |-- [     937710]  bst_lr_nought_3_400_round.dump
|       |-- [     923222]  bst_lr_nought_3_400_round.save
|       |-- [     535196]  bst_lr_nought_3.dump
|       `-- [     520798]  bst_lr_nought_3.save
|-- [       4096]  notebooks
|   |-- [    5031035]  co_occurrence_analysis.ipynb
|   |-- [      17536]  duplicates.ipynb
|   |-- [      43886]  embeddings_with_linear_svc_eval_pipeline.ipynb
|   |-- [   16133088]  exploratory_data_analysis.ipynb
|   |-- [      15852]  feature_creation_with_large_language_models.ipynb
|   |-- [      50916]  fine_tuning_transformers.ipynb
|   |-- [     226962]  initial_data_analysis.ipynb
|   |-- [     700731]  initial_data_exploration.ipynb
|   |-- [    3828887]  lightgbm.ipynb
|   |-- [      62326]  meta_embeddings.ipynb
|   |-- [     547534]  meta_feature_exploration.ipynb
|   |-- [      56666]  model_pipeline_CountVectorizer.ipynb
|   |-- [     363613]  model_pipeline_tfidfVectorizer.ipynb
|   |-- [       7000]  pipeline_spacy_VectorTransformer.ipynb
|   |-- [   15645857]  presentation.ipynb
|   |-- [       3911]  template_model_pipeline.ipynb
|   |-- [       9530]  template_model_tutorial.ipynb
|   |-- [   12594651]  topic_modeling.ipynb
|   |-- [      63956]  tweet_sentiment.ipynb
|   `-- [       4096]  wandb
|       |-- [         52]  debug-internal.log -> run-20210111_171912-1phci8oe/logs/debug-internal.log
|       |-- [         43]  debug.log -> run-20210111_171912-1phci8oe/logs/debug.log
|       |-- [         28]  latest-run -> run-20210111_171912-1phci8oe
|       `-- [       4096]  run-20210111_171912-1phci8oe
|           |-- [       4096]  files
|           |   |-- [        176]  config.yaml
|           |   |-- [     119781]  output.log
|           |   |-- [       2623]  requirements.txt
|           |   |-- [        765]  wandb-metadata.json
|           |   `-- [        156]  wandb-summary.json
|           |-- [       4096]  logs
|           |   |-- [   14152320]  debug-internal.log
|           |   `-- [       6527]  debug.log
|           `-- [    8212534]  run-1phci8oe.wandb
|-- [       6700]  README.md
|-- [       4096]  references
|   `-- [          0]  literature.bib
|-- [       4096]  reports
|   |-- [       4096]  data
|   |   |-- [       3312]  meta_features_individual.pck
|   |   `-- [     915251]  meta_features_union.pck
|   |-- [       4096]  figures
|   |   |-- [     148066]  ablaze.png
|   |   |-- [     159152]  ModalNet-21.png
|   |   |-- [     155087]  PLMfamily.jpg
|   |   |-- [     188129]  transformer.png
|   |   `-- [     361452]  twitter.png
|   `-- [       4096]  gephi_reports
|       |-- [       4096]  degree
|       |   |-- [      12439]  degree-distribution.png
|       |   `-- [        154]  report.html
|       |-- [       4096]  diameter
|       |   |-- [      10828]  Betweenness Centrality Distribution.png
|       |   |-- [      12389]  Closeness Centrality Distribution.png
|       |   |-- [       9938]  Eccentricity Distribution.png
|       |   |-- [      12727]  Harmonic Closeness Centrality Distribution.png
|       |   `-- [        662]  report.html
|       |-- [       4096]  exported
|       |   `-- [    2014391]  no_labels.png
|       |-- [       4096]  HITS
|       |   |-- [      10870]  authorities.png
|       |   |-- [      10327]  hubs.png
|       |   `-- [        366]  report.html
|       |-- [       4096]  modularity_10
|       |   |-- [      15064]  communities-size-distribution.png
|       |   `-- [        715]  report.html
|       |-- [       4096]  modularity_5
|       |   |-- [      15921]  communities-size-distribution.png
|       |   `-- [        715]  report.html
|       |-- [       4096]  Project1
|       |   |-- [       2555]  about.html
|       |   |-- [       1634]  estadisticas.json
|       |   |-- [       4096]  img
|       |   |   |-- [       2595]  download.png
|       |   |   |-- [      12799]  glyphicons-halflings.png
|       |   |   |-- [       4176]  loading.gif
|       |   |   |-- [      21384]  logo_final.png
|       |   |   |-- [        327]  opendata.png
|       |   |   |-- [      22475]  upm.png
|       |   |   `-- [     142574]  utpl.jpg
|       |   |-- [       5037]  index.html
|       |   |-- [       5187]  info.html
|       |   |-- [       4096]  js
|       |   |   |-- [      28538]  bootstrap.min.js
|       |   |   |-- [      93636]  jquery-1.8.3.min.js
|       |   |   |-- [       4341]  jquery.cookie.js
|       |   |   |-- [     110593]  jquery-ui-1.10.3.custom.min.js
|       |   |   |-- [      12669]  loxawebsite-0.9.1.js
|       |   |   |-- [      30907]  sigma.min.js
|       |   |   `-- [       6173]  sigma.parseGexf.js
|       |   |-- [       4096]  styles
|       |   |   |-- [     121689]  bootstrap.css
|       |   |   |-- [       4096]  images
|       |   |   |   |-- [       1738]  animated-overlay.gif
|       |   |   |   |-- [        418]  ui-bg_diagonals-thick_18_b81900_40x40.png
|       |   |   |   |-- [        312]  ui-bg_diagonals-thick_20_666666_40x40.png
|       |   |   |   |-- [        205]  ui-bg_flat_10_000000_40x100.png
|       |   |   |   |-- [        262]  ui-bg_glass_100_f6f6f6_1x400.png
|       |   |   |   |-- [        348]  ui-bg_glass_100_fdf5ce_1x400.png
|       |   |   |   |-- [        207]  ui-bg_glass_65_ffffff_1x400.png
|       |   |   |   |-- [       5815]  ui-bg_gloss-wave_35_f6a828_500x100.png
|       |   |   |   |-- [        278]  ui-bg_highlight-soft_100_eeeeee_1x100.png
|       |   |   |   |-- [        328]  ui-bg_highlight-soft_75_ffe45c_1x100.png
|       |   |   |   |-- [       6922]  ui-icons_222222_256x240.png
|       |   |   |   |-- [       4549]  ui-icons_228ef1_256x240.png
|       |   |   |   |-- [       4549]  ui-icons_ef8c08_256x240.png
|       |   |   |   |-- [       4549]  ui-icons_ffd27a_256x240.png
|       |   |   |   `-- [       6299]  ui-icons_ffffff_256x240.png
|       |   |   `-- [      18120]  jquery-ui-1.10.3.custom.min.css
|       |   `-- [       4096]  Workspace1
|       |       |-- [    9012123]  Workspace1.csv
|       |       |-- [    2353376]  Workspace1.gexf
|       |       `-- [     202070]  Workspace1.pdf
|       |-- [       4096]  screenshots
|       |   |-- [    1567191]  network.png
|       |   `-- [      66383]  screenshot_142429.png
|       `-- [       4096]  weighted_degree
|           |-- [        174]  report.html
|           `-- [      15232]  w-degree-distribution.png
`-- [       4096]  src
    |-- [       8099]  evaluation.py
    |-- [       4096]  features
    |   |-- [      24766]  meta_features_spacy.py
    |   `-- [       4096]  __pycache__
    |       `-- [      33319]  meta_features_spacy.cpython-38.pyc
    |-- [          0]  __init__.py
    `-- [       4096]  __pycache__
        |-- [       4452]  evaluation.cpython-36.pyc
        `-- [       4478]  evaluation.cpython-38.pyc

50 directories, 187 files


```
</p>
</details>

<details>
<summary><b>Milestones</b>
</summary>
<p>

1. MS: Agree on core technologies & frameworks
    * **Description:**
        - Programming: Python3, scikit-learn, sphinx
        - Infrastructure: github, docker, notebooks
        - orga: discord, BBB
    * **Tasks:**
        - [x] Supervisor Kick-Off: everyone
        - [x] Discord server setup: everyone
        - [x] Repo setup: Chris
        - [x] Project plan: Julian + Karl
    * **Deliverables:** Project Plan, repository
    * **Due:** 23.11.2020

2. MS: Implement evaluation pipeline:
    * **Description:**
        - Shared evaluation pipeline: (1) read data -> (2) feed to model -> (3) generate score, plots + submission file
        - Idea: Once pipeline stands, everyone plays with models in step (2)
    * **Tasks:**
        - [x] Implement data assembly
        - [x] Implement K-Fold Cross-Validation
        - [x] Generate result graphs and scores
        - [x] Generate submission file
    * **Deliverables:** Shared Evaluation Pipeline
3. MS: First Model Iteration:
    * **Description:**
        - Baseline model: Integrate simple model from tutorial (https://www.kaggle.com/philculliton/nlp-getting-started-tutorial)
        - Everyone plays with models in step (2)
    * **Tasks:**
        - [x] Implement model from tutorial
        - [x] Implement own features & models
    * **Deliverables:** Model dumps and their evaluation results
4. MS: Second Model Iteration:
    * **Description:**
        - Sync: Share insights & features from first iteration in group and with supervisor
        - Everyone attempts to improve their models from insights & features
    * **Tasks:**
        - [x] Sync Meeting (with supervisor?): everyone (pending)
        - [x] Improve own features & models: everyone (pending)
    * **Deliverables:** Model dumps and their evaluation results
5. MS: Final Presentation
    * **Tasks:**
        - [x] Intro slides
        - [x] Leaderboard stats slide
        - [x] Our result slides
    * **Deliverables:** Presentation Slides
    * **Due:** ? (end of january)
6. MS: Final Report
    * **Tasks:**
        - [ ] LaTeX template ? (pending)
        - [ ] Abstract: ? (pending)
        - [ ] Intro: ? (pending)
        - [ ] Related Work: ? (pending)
        - [ ] Methods: ? (pending)
        - [ ] Result: ? (pending)
        - [ ] Discussion: ? (pending)
    * **Deliverables:** Report Document
    * **Due:** ? (end of february)
</p>
</details>

## Programming Environment
### Makefile
#### Regular Image:
```console
foo@bar:/disaster-tweets$ make docker-build-image
```
```console
foo@bar:/disaster-tweets$ make docker-run-jupyter
```

#### GPU Image:
```console
foo@bar:/disaster-tweets$ make docker-build-image-gpu
```
```console
foo@bar:/disaster-tweets$ make docker-run-jupyter-gpu
```
#### NVIDIA RAPIDS Image:
```console
foo@bar:/disaster-tweets$ make docker-run-rapids
```
### docker-compose
To fire up the environment using docker-compose, run:
```console
foo@bar:/disaster-tweets$ docker-compose up
```



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
