# dcase2026\_task2\_evaluator
The **dcase2026\_task2\_evaluator** is a script for calculating the AUC, pAUC, precision, recall, and F1 scores from the anomaly score list for the [evaluation dataset](https://zenodo.org/records/20437238) in DCASE 2026 Challenge Task 2 "Noise-aware Unsupervised Anomalous Sound Detection for Machine Condition Monitoring"

[https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)

## Description

The **dcase2026\_task2\_evaluator** consists of two scripts:

- `dcase2026_task2_evaluator.py`
    - This script outputs the AUC and pAUC scores by using:
      - Ground truth of the normal and anomaly labels
      - Anomaly scores for each wave file listed in the csv file for each machine type, section, and domain
      - Detection results for each wave file listed in the csv file for each machine type, section, and domain
- `03_evaluation_eval_data.sh`
    - This script execute `dcase2026_task2_evaluator.py`.

## Usage
### 1. Clone repository
Clone this repository from Github.

### 2. Prepare data
- Anomaly scores
    - Generate csv files `anomaly_score_<machine_type>_section_<section_index>_test.csv` and `decision_result_<machine_type>_section_<section_index>_test.csv` or `anomaly_score_DCASE2026T2<machine_type>_section_<section>_test_seed<seed><tag>_Eval.csv` and `decision_result_DCASE2026T2<machine_type>_section_<section>_test_seed<seed><tag>_Eval.csv` by using a system for the [evaluation dataset](https://zenodo.org/record/20437238). (The format information is described [here](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#submission).)
- Rename the directory containing the csv files to a team name
- Move the directory into `./teams/`

### 3. Check directory structure
- ./dcase2026\_task2\_evaluator
    - /dcase2026\_task2\_evaluator.py
    - /03\_evaluation\_eval\_data.sh
    - /ground\_truth\_attributes
        - ground\_truth\_BlowerDustCollector\_section\_00\_test.csv
        - ground\_truth\_Sander\_section\_00\_test.csv
        - ...
    - /ground\_truth\_data
        - ground\_truth\_BlowerDustCollector\_section\_00\_test.csv
        - ground\_truth\_Sander\section\_section\_00\_test.csv
        - ...
    - /ground\_truth\_domain
        - ground\_truth\_BlowerDustCollector\_section\_00\_test.csv
        - ground\_truth\_Sander\_section\_00\_test.csv
        - ...
    - /teams
        - /\<team\_name\_1\>
            - /\<system\_name\_1\>
                - anomaly\_score\_BlowerDustCollector\_section\_00\_test.csv
                - anomaly\_score\_Sander\_section\_00\_test.csv
                - ...
                - decision\_result\_ToothBrush\_section\_00\_test.csv
                - decision\_result\_ToyDrone\_section\_00\_test.csv
            - /\<system\_name\_2\>
                - anomaly\_score\_DCASE2026T2BlowerDustCollector\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
                - anomaly\_score\_DCASE2026T2Sander\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
                - ...
                - decision\_result\_DCASE2026T2ToothBrush\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
                - decision\_result\_DCASE2026T2ToyDrone\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
        - /\<team\_name\_2\>
            - /\<system\_name\_3\>
                - anomaly\_score\_BlowerDustCollector\_section\_00\_test.csv
                - anomaly\_score\_Sander\_section\_00\test.csv
                - ...
                - decision\_result\_ToothBrush\_section\_00\_test.csv
                - decision\_result\_ToyDrone\_section\_00\_test.csv
        - ...
    - /teams\_result
        - \<system\_name\_1\>\_result.csv
        - \<system\_name\_2\>\_result.csv
        - \<system\_name\_3\>_result.csv
        - ...
    - /teams\_additional\_result \*`out_all==True`
        - teams\_official\_score.csv
        - teams\_official\_score\_paper.csv
        - teams\_section\_00\_auc.csv
        - teams\_section\_00\_score.csv
        - /\<system\_name\_1\>
            - official\_score.csv
            - \<system\_name\_1\>\_BlowerDustCollector\_section\_00\_anm\_score.png
            - ...
            - \<system\_name\_1\>\_ToyDrone\_section\_00\_anm\_score.png
        - /\<system\_name\_2\>
            - official\_score.csv
            - \<system\_name\_2\>\_BlowerDustCollector\_section\_00\_anm\_score.png
            - ...
            - \<system\_name\_2\>\_ToyDrone\_section\_00\_anm\_score.png
        - /\<system\_name\_3\>
            - official\_score.csv
            - \<system\_name\_3\>\_BlowerDustCollector\_section\_00\_anm\_score.png
            - ...
            - \<system\_name\_3\>\_ToyDrone\_section\_00\_anm\_score.png
        - ...
    - /tools
        - plot\_anm\_score.py
        - test\_plots.py
    - /README.md


### 4. Change parameters
The parameters are defined in the script `dcase2026_task2_evaluator.py` as follows.
- **MAX\_FPR**
    - The FPR threshold for pAUC : default 0.1
- **--result\_dir**
    - The output directory : default `./teams_result/`
- **--teams\_root\_dir**
    - Directory containing team results. : default `./teams/`
- **--dir\_depth**
    - What depth to search `--teams_root_dir` using glob. : default `2`
    - If --dir\_depth=2, then `glob.glob(<teams_root_dir>/*/*)`
- **--tag**
    - File name tag. : default `_id(0_)`
    - If using filename is DCASE2026 baseline style, change parameters as necessary. 
- **--seed**
    - Seed used during train. : default `13711`
    - If using filename is DCASE2026 baseline style, change parameters as necessary.
- **--out\_all**
    - If this parameter is `True`, export supplemental data. : default `False`
- **--additional\_result\_dir**
    - The output additional results directory. : default `./teams_additional_result/`
    - Used when `--out_all==True`.

### 5. Run script
Run the script `dcase2026_task2_evaluator.py`
```
$ python dcase2026_task2_evaluator.py
```
or
```
$ bash 03_evaluation_eval_data.sh
```
The script `dcase2026_task2_evaluator.py` calculates the AUC, pAUC, precision, recall, and F1 scores for each machine type, section, and domain and output the calculated scores into the csv files (`<system_name_1>_result.csv`, `<system_name_2>_result.csv`, ...) in **--result\_dir** (default: `./teams_result/`).
If **--out\_all=True**, each team results are then aggregated into a csv file (`teams_official_score.csv`, `teams_official_score_paper.csv`) in **--additional\_result\_dir** (default: `./teams_additional_result`).

### 6. Check results
You can check the AUC, pAUC, precision, recall, and F1 scores in the `<system_name_N>_result.csv` in **--result\_dir**.
The AUC, pAUC, precision, recall, and F1 scores for each machine type, section, and domain are listed as follows:

`<section_name_N>_result.csv`

```csv
BlowerDustCollector
section,AUC (all),AUC (source),AUC (target),pAUC,precision (source),precision (target),recall (source),recall (target),F1 score (source),F1 score (target)
00,0.7893000000000001,0.6948000000000001,0.8837999999999999,0.6178947368421053,0.5157894736842106,0.5681818181818182,0.98,1.0,0.6758620689655174,0.7246376811594203
,,AUC,pAUC,precision,recall,F1 score
arithmetic mean,,0.7893,0.6178947368421053,0.5419856459330143,0.99,0.7002498750624688
harmonic mean,,0.7779858608893957,0.6178947368421053,0.5407194879717503,0.98989898989899,0.6994005138452756
source harmonic mean,,0.6948000000000001,0.6178947368421053,0.5157894736842106,0.98,0.6758620689655174
target harmonic mean,,0.8837999999999999,0.6178947368421053,0.5681818181818182,1.0,0.7246376811594204

...

ToyDrone
section,AUC (all),AUC (source),AUC (target),pAUC,precision (source),precision (target),recall (source),recall (target),F1 score (source),F1 score (target)
00,0.5903,0.7126000000000001,0.468,0.5510526315789473,0.5853658536585366,0.524390243902439,0.48,0.86,0.5274725274725274,0.6515151515151516
,,AUC,pAUC,precision,recall,F1 score
arithmetic mean,,0.5903,0.5510526315789473,0.5548780487804879,0.6699999999999999,0.5894938394938395
harmonic mean,,0.5649615449771302,0.5510526315789473,0.5532028946663093,0.6161194029850746,0.582968507272984
source harmonic mean,,0.7126000000000001,0.5510526315789473,0.5853658536585366,0.48,0.5274725274725274
target harmonic mean,,0.468,0.5510526315789473,0.524390243902439,0.86,0.6515151515151516

,,AUC,pAUC,precision,recall,F1 score
"arithmetic mean over all machine types, sections, and domains",,0.6367,0.5713684210526317,0.5332265500685649,0.852,0.6463695738181222
"harmonic mean over all machine types, sections, and domains",,0.6138396081093476,0.568731371472361,0.5314162750580497,0.8108326266667873,0.6420413585067419
"source harmonic mean over all machine types, sections, and domains",,0.6606242657626471,0.568731371472361,0.5371219163581458,0.7326726867597045,0.6198397073185856
"target harmonic mean over all machine types, sections, and domains",,0.5732431794118468,0.568731371472361,0.5258305773466164,0.9076598155637651,0.6658925476064297

official score,,0.5980289616592438
official score ci95,,2.6619662647486664e-05
```

Aggregated results for each baseline are listed as follows:

```_seed13711_official_score_paper.csv
System,metric,h-mean,a-mean,ToyDrone,ToothBrush,SewingMachine,BlowerDustCollector,Sander
baseline_MAHALA,AUC (source),0.6085641541440189,0.62908,0.7275999999999999,0.4398,0.6033999999999999,0.6766,0.698
baseline_MAHALA,AUC (target),0.4985220763401413,0.53948,0.5428,0.394,0.43760000000000004,0.872,0.45100000000000007
baseline_MAHALA,"pAUC (source, target)",0.5467370545597742,0.5519999999999999,0.5926315789473684,0.5278947368421053,0.5057894736842106,0.6389473684210526,0.49473684210526314
baseline_MAHALA,TOTAL score,0.5476277174340481,0.57352,,,,,
baseline_MSE,AUC (source),0.6606242657626471,0.6634800000000001,0.7126000000000001,0.5902,0.6484,0.6948000000000001,0.6714
baseline_MSE,AUC (target),0.5732431794118468,0.60992,0.468,0.6434000000000001,0.6142,0.8837999999999999,0.4402
baseline_MSE,"pAUC (source, target)",0.568731371472361,0.5713684210526317,0.5510526315789473,0.5978947368421053,0.5810526315789474,0.6178947368421053,0.5089473684210526
baseline_MSE,TOTAL score,0.5980289616592438,0.614922807017544,,,,,

```

## License 
This project is licensed under the terms described in [LICENSEv2.1.pdf](LICENSEv2.1.pdf).

## Citation

If you use this system, please cite all the following four papers:

+ Tomoya Nishida, Noboru Harada, Daiki Takeuchi, Daisuke Niizumi, Keisuke Imoto, Kota Dohi, Harsh Purohit, Takashi Endo, and Yohei Kawaguchi. Description and discussion on DCASE 2026 challenge task 2: noise-aware unsupervised anomalous sound detection for machine condition monitoring. In arXiv e-prints: 2606.01578, 2026. [URL](https://arxiv.org/abs/2606.01578)
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito. ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions. In Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), 1–5. Barcelona, Spain, November 2021. [URL](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Harada_6.pdf)
+ Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi. MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task. In Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022). Nancy, France, November 2022. [URL](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Dohi_62.pdf)
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, and Masahiro Yasuda. First-shot anomaly detection for machine condition monitoring: a domain generalization baseline. Proceedings of 31st European Signal Processing Conference (EUSIPCO), pages 191–195, 2023. [URL](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000191.pdf)
