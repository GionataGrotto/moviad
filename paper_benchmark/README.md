# Benchmark audio del paper

Questa cartella contiene gli script per eseguire i modelli audio e raccogliere le performance in modo ripetibile, fuori dalla libreria `moviad`.

Gli script scrivono i risultati in:

```text
results/paper_benchmark/
```

Dataset supportati:

- `MIMII`, per anomaly detection industriale audio.
- `EnvMix`, costruito da UrbanSound8K come background e ESC-50 come anomalie.

Modelli supportati:

- `patchcore`
- `padim`
- `cfa`
- `stfpm`
- `dinomaly`
- `draem`

Metriche principali:

- MIMII: `img_roc_auc`, `f1_img`, `pr_auc_img`.
- EnvMix: metriche image-level, spectrogram-level e temporal-level quando disponibili.

## 1. Preparare il PC

Dalla root della repo `moviad`:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

Se usi `uv`, va bene anche:

```powershell
uv sync
```

ma non e' obbligatorio.

## 2. Aggiungere il checkpoint CLAP

Per usare il backbone grande `HTSAT-base`, installa inoltre la dipendenza
ufficiale LAION:

```powershell
uv pip install -r requirements-htsat.txt
```

Il checkpoint grande supportato dal loader è:

```text
src/moviad/weights/music_speech_audioset_epoch_15_esc_89.98.pt
```

Nel file di configurazione usa `backbone: "HTSAT-base"` e layer come
`["0", "1", "2", "3"]`. Il checkpoint Cnn14 originale continua a funzionare
con `backbone: "Cnn14"`.

Per ottenere performance sensate devi usare il backbone pre-addestrato. Puoi
mettere il checkpoint qui:

```text
src/moviad/weights/clap_encoder.pth
```

In alternativa, il checkpoint può restare fuori dalla repo e si può indicare il
percorso nella configurazione, senza inserirlo nel repository:

```json
"clap_checkpoint": "C:/Users/ggion/Downloads/clap_encoder.pth"
```

Se la cartella `weights` non esiste, creala:

```powershell
New-Item -ItemType Directory src\moviad\weights
```

Solo per verificare che la pipeline funzioni senza checkpoint puoi mettere nella config:

```json
"pretrained": false
```

Questa modalita' usa pesi random e non va usata per il report finale.

## 3. Preparare la config locale

Copia la config di esempio:

```powershell
Copy-Item paper_benchmark\config.example.json paper_benchmark\config.local.json
```

Poi modifica `paper_benchmark\config.local.json`.

Imposta almeno questi campi:

```json
{
  "device": "auto",
  "pretrained": true,
  "mimii": {
    "dataset_path": "C:/path/al/dataset/mimii"
  },
  "envmix": {
    "urban_path": "C:/path/a/UrbanSound8K",
    "esc50_path": "C:/path/a/ESC-50"
  }
}
```

Valori utili per `device`:

- `auto`: usa CUDA se disponibile, altrimenti CPU.
- `cuda`: forza GPU.
- `cpu`: forza CPU.

## 4. Test rapido senza dataset

Prima controlla che import e struttura siano ok:

```powershell
python paper_benchmark\run_smoke.py
```

Se stampa `Smoke OK`, la repo e' importabile correttamente.

## 5. Primo run piccolo su MIMII

Esegui prima un run debug, cosi' controlli path, checkpoint e memoria senza aspettare troppo:

```powershell
python paper_benchmark\run_mimii.py --config paper_benchmark\config.local.json --debug --methods patchcore
```

Se funziona, puoi provare tutti i modelli in debug:

```powershell
python paper_benchmark\run_mimii.py --config paper_benchmark\config.local.json --debug --methods patchcore padim cfa stfpm
```

## 6. Run completo su MIMII

Quando il debug e' ok:

```powershell
python paper_benchmark\run_mimii.py --config paper_benchmark\config.local.json --methods patchcore padim cfa stfpm
```

I risultati vengono salvati in:

```text
results/paper_benchmark/mimii_results.csv
```

Ogni riga contiene metodo, categoria, SNR, machine id, seed e metriche.

## 7. Primo run piccolo su EnvMix

Anche per EnvMix parti dal debug:

```powershell
python paper_benchmark\run_envmix.py --config paper_benchmark\config.local.json --debug --methods patchcore
```

Poi prova tutti i modelli in debug:

```powershell
python paper_benchmark\run_envmix.py --config paper_benchmark\config.local.json --debug --methods patchcore padim cfa stfpm
```

## 8. Run completo su EnvMix

Quando il debug e' ok:

```powershell
python paper_benchmark\run_envmix.py --config paper_benchmark\config.local.json --methods patchcore padim cfa stfpm
```

I risultati vengono salvati in:

```text
results/paper_benchmark/envmix_results.csv
```

## 9. Generare il report finale

Dopo aver eseguito MIMII e/o EnvMix:

```powershell
python paper_benchmark\report.py --results results\paper_benchmark
```

Il report viene scritto qui:

```text
results/paper_benchmark/report.md
```

Il report aggrega le righe dei CSV e mostra media +/- deviazione standard per gruppo.

## 10. Faithfulness su EnvMix

Gli script EnvMix calcolano anche ff_v1 e ff_v2 per CFA, PaDiM, PatchCore e STFPM. La soglia di FF v2 e configurabile nel campo faithfulness.v2_threshold della config.

I valori vengono salvati in envmix_results.csv e inclusi nel report aggregato.
## 11. Ordine consigliato

Esegui in questo ordine:

```powershell
python paper_benchmark\run_smoke.py
python paper_benchmark\run_mimii.py --config paper_benchmark\config.local.json --debug --methods patchcore
python paper_benchmark\run_envmix.py --config paper_benchmark\config.local.json --debug --methods patchcore
python paper_benchmark\run_mimii.py --config paper_benchmark\config.local.json --methods patchcore padim cfa stfpm
python paper_benchmark\run_envmix.py --config paper_benchmark\config.local.json --methods patchcore padim cfa stfpm
python paper_benchmark\report.py --results results\paper_benchmark
```

## 12. DCASE 2026 Task 2

Il runner usa il solo canale 0 (microfono vicino), addestra un modello
separato per macchina sui clip normali `train` e genera gli output compatibili
con l'evaluator vendorizzato in `dcase2026_task2_evaluator`.

Imposta il percorso in `paper_benchmark\config.local.json`:

```json
{
  "dcase2026": {
    "dataset_path": "C:/path/to/Datasets/DCASE"
  }
}
```

Poi esegui:

```powershell
python paper_benchmark\run_dcase2026_task2.py `
  --config paper_benchmark\config.local.json `
  --methods patchcore padim stfpm cfa dinomaly draem
```

Per un controllo rapido:

```powershell
python paper_benchmark\run_dcase2026_task2.py `
  --methods patchcore --debug --epochs 1
```

`--dataset-path` resta disponibile come override temporaneo del valore nel
config, ma non è obbligatorio.

Gli score e le decisioni vengono scritti in
`dcase2026_task2_evaluator\dev\teams\moviad\<method>`. Il risultato aggregato
contiene `AUC (all)` (source+target), `AUC (source)` e `AUC (target)`.
L'evaluator può essere eseguito dalla sua directory con
`--ground_truth_root dev`.

Per DCASE il file-level score usa di default la media temporale dei top-5
valori in frequenza (`temporal_topk_mean`). L'opzione è indipendente da MIMII e
può essere cambiata anche da riga di comando:

```bash
python paper_benchmark/run_dcase2026_task2.py \
  --config paper_benchmark/config.local.json \
  --methods padim \
  --score-aggregation max
```

Sono disponibili `max`, `mean` e `temporal_topk_mean`; per quest'ultima si può
impostare il numero di valori con `--score-topk`.

### Griglia di esperimenti sui metodi standard (DCASE)

`run_dcase_grid.py` parte dalla baseline (Cnn14 `conv_block2-4`, microfono vicino, training solo source)
e cambia un solo fattore per volta, sempre con test sui clip source. Esperimenti:
`layers_fine` (`conv_block1-3`), `layers_deep` (`conv_block3-5`), `train_all` (aggiunge i 10 clip target),
`near_minus_far` (canale 0 - canale 1), `stfpm_random_student` (studente a pesi casuali, come nel paper
STFPM), `patchcore_standardize` (canali standardizzati), `patchcore_big_bank` (60000 patch),
`padim_full_cov`, `cfa_fine_memory` (`gamma_c=1`), `dinomaly_small`, `dinomaly_base` ed `epochs_long`
(100 epoche per CFA, STFPM e Dinomaly), `spec_band_norm` (log-mel standardizzato per banda mel con le
statistiche del training), `spec_fill_dead_bands` (bande mel sopra 8 kHz, vuote a 16 kHz, sostituite con l'ultima
banda viva), `spec_both` e `agg_mean` (punteggio = media della mappa invece del top-k). Ogni esperimento gira solo sui metodi a cui si applica. Ogni run e'
isolato in `dcase2026_task2_evaluator/dev/grid_<nome>` e la tabella riassuntiva (AUC source media per
esperimento e metodo) viene riscritta in `results/paper_benchmark/grid_summary.md` dopo ogni esperimento.

`beats_late` e `beats_mid` usano BEATs come backbone di PatchCore, PaDiM e CFA (STFPM e Dinomaly esclusi):
i token del ViT (griglia tempo x 8 bande) diventano le mappe di feature. Servono i sorgenti BEATs e il
checkpoint, indicati con `beats_checkpoint` e `beats_code_dir` nel config (o con le chiavi `knn_beats_*` del
config kNN). `layers` sono numeri di blocco da 1 a 12. Il codice assume la struttura dei sorgenti unilm
(blocchi con tempo in prima dimensione, token ordinati per tempo e poi banda): se al primo run da' un errore
di forma, e' il primo posto dove guardare.

Per cercare la configurazione migliore con BEATs ci sono tre gruppi, da lanciare con `--preset` in
quest'ordine: `beats_layers` (quali blocchi: 8-10-12, 4-6-8, solo 12, solo 10, 10-11-12, 6-9-12),
`beats_protocol` (+ i 10 clip target, near - far, punteggio = media della mappa) e `beats_heads`
(PaDiM con covarianza completa o 576 canali, PatchCore con memoria da 60000, CFA con `gamma_c=1`);
`beats_search` li lancia tutti. In fondo a `grid_summary.md` la tabella indica il migliore per metodo.
Per confermare la scelta finale ripeti il migliore con altri seed (`--seed`, `--run-suffix`).

**Idee per banda e aggregazioni** (preset `band_search`; `band_full` aggiunge i riferimenti rilanciati con lo
stesso codice). Si basano sul fatto che per l'audio conta la banda di frequenza, non il tempo assoluto:
- `band_pc_banks`: PatchCore con una memoria per banda (la memoria totale e' divisa tra le bande), cosi' una
  patch si confronta solo con patch normali della propria banda.
- `band_pc_std`, `band_pc_banks_std`: standardizzazione dei punteggi per banda con media e deviazione di clip
  normali tenuti da parte (il 10% dei batch di training, esclusi dalla memoria: i punteggi dei clip della
  memoria sono quasi zero e falserebbero le statistiche; `score_std_holdout` cambia la quota).
- `band_padim_gauss`, `band_padim_gauss_full`, `band_padim_gauss_full_std`: PaDiM con una gaussiana per banda
  condivisa da tutti i frame (invece di una per posizione tempo-frequenza): 990 volte piu' campioni per
  gaussiana, covarianza completa ben condizionata, clip di qualsiasi durata. La covarianza completa e' l'unica
  variante che aveva aiutato PaDiM.
- `dinomaly_l2`: Dinomaly con distanza L2 al posto del coseno tra feature di encoder e decoder.
- `--aggregation-sweep` calcola dallo stesso modello addestrato anche `max`, `mean`, `top5pct` e `top10pct`
  (media del 5% / 10% dei valori piu' alti della mappa); compaiono come colonne `metodo@aggregazione` nel
  riepilogo. Si possono scegliere altre con `--extra-aggregations` (qualsiasi `top<N>pct`).

```bash
uv run python paper_benchmark/run_dcase_grid.py --base-config paper_benchmark/config.knn.json --methods patchcore padim --preset band_full --aggregation-sweep
```

**STFPM** (preset `stfpm_search`, da usare con `--methods stfpm`). Il trainer usava sempre SGD con learning
rate 0,4, la scelta del paper per studenti CNN su immagini; ora `stfpm_optimizer` (`sgd` o `adamw`) e
`stfpm_lr` sono nel config. Esperimenti: `baseline` (riferimento), `stfpm_sgd_lr0p1`, `stfpm_sgd_lr0p02`,
`stfpm_adamw_1e4` (studente Cnn14 con learning rate piu' bassi), `layers_fine`, `layers_deep`, e con BEATs come
backbone lo studente pre-addestrato con AdamW (`stfpm_beats_late_adamw_1e5`, `stfpm_beats_late_adamw_1e4`,
`stfpm_beats_b12_adamw_1e5`: il learning rate di 0,4 farebbe divergere un ViT).

```bash
uv run python paper_benchmark/run_dcase_grid.py --base-config paper_benchmark/config.knn.json --methods stfpm --preset stfpm_search --aggregation-sweep --skip-done
```

**Tutto in un comando** (`run_all.py`). Esegue in sequenza ogni coppia (esperimento, metodo) di `PLAN`
(riferimenti BEATs, 3 seed in piu' del CFA, `handcrafted`, canale `ls`/`ls_half`, STFPM, `knn_ft`), ciascuna in un
processo separato: se una salta viene ritentata (`--retries`, default 1) e poi si passa alla successiva. I fallimenti
sono in `results/paper_benchmark/all_failures.json`. Le coppie gia' complete sono saltate, quindi lo stesso comando
riprende un job interrotto; `--only-failed` rilancia solo i falliti, `--only <testo>` filtra, `--list` mostra il piano.
A fine corsa calcola le fusioni (`fuse_scores.py`, rank medio per macchina) in `results/paper_benchmark/fusion_dcase2026.md`.

```bash
uv run python paper_benchmark/run_all.py --base-config paper_benchmark/config.knn.json
```

Nuovi metodi/canali: `handcrafted` (statistiche per sotto-banda + Mahalanobis con shrinkage, nessun modello),
`knn_ft` (BEATs con LoRA sulle proiezioni q/v e ArcFace sugli attributi nei nomi dei file; per le macchine senza
attributi, `noAttribute`, l'encoder resta invariato; chiavi `finetune_*`), `dcase2026.channel` = `ls` / `ls_half`
(near meno la predizione lineare ai minimi quadrati dal far mic; rischio sulle macchine Emu, dove il far e' il near filtrato).
`fuse_scores.py --inputs nome=cartella ...` fonde cartelle di punteggi gia' esistenti.

Aggregazioni del punteggio di clip: `max`, `mean`, `temporal_topk_mean` (per frame la media delle 5 bande piu' alte,
poi media su tutti i frame), `top<N>pct` (N% delle patch piu' alte dell'intera mappa) e `frames<N>pct` (come
`temporal_topk_mean` ma media solo dell'N% di frame piu' alti). Nelle fusioni di `run_all.py` un elemento puo' essere una
lista di run (es. i seed), mediati prima e contati come un solo componente.

Canali LS (`dcase2026.channel`): `ls` (un filtro per clip), `ls_half`, `ls_win` (filtro su blocchi di ~2 s, segue un rumore
che cambia), `ls_global` (un filtro per macchina stimato sui clip di training). `handcrafted`: `handcrafted_stats`
(`mean`, `std`, `diffstd`, `kurtosis`, `q10`, `q90`, `corr`), `handcrafted_scorer` (`mahalanobis`, `knn`, `iforest`).
`knn_ft`: `finetune_pseudo_classes` = k cluster k-means come classi quando i nomi dei file non hanno attributi
(fan, ToyCar). `error_analysis.py --scores <cartella> --machine <m> --attributes <ground_truth_attributes>` mostra
AUC per attributo, anomalie mancate e falsi allarmi.

`diagnose_signal_features.py --dataset-path <DCASE>` calcola per macchina l'AUC di semplici statistiche
del segnale (RMS, centroide spettrale, piattezza, quota di alte frequenze): serve a capire se un'inversione
sistematica (AUC sotto 0,5) dipende dall'energia del clip.

```bash
uv run python paper_benchmark/run_dcase_grid.py --base-config paper_benchmark/config.knn.json
```

Opzioni utili: `--methods patchcore padim`, `--experiments baseline layers_deep`, `--epochs 10`
(CFA, STFPM, Dinomaly), `--dry-run` per vedere i comandi. I metodi senza layer (Dinomaly) saltano gli
esperimenti sui layer. Il config di base serve per `dcase2026.dataset_path` e il device.

### kNN su embedding BEATs (DCASE)

Metodo `knn`: embedding clip-level di un encoder pre-addestrato (media nel tempo del
blocco transformer scelto) e punteggio = distanza dal vicino piu' prossimo nei clip
di training. E' la famiglia "embedding pre-addestrati + outlier detection".
Servono due file che non sono nel repo: i sorgenti BEATs (`BEATs.py` di unilm) e un
checkpoint come `BEATs_iter3_plus_AS2M.pt`. L'audio viene caricato a 16 kHz.

```bash
python paper_benchmark/run_dcase2026_task2.py   --config paper_benchmark/config.dcase2026_knn_beats.example.json   --methods knn
```

`knn_layers` indica i blocchi transformer con numerazione da 1 (12 = ultimo di
BEATs-base); con piu' layer le distanze vengono mediate. `knn_metric` e' `cosine`
o `euclidean`, `knn_top_k` la media sui k vicini. Il modello non ha mappe
tempo-frequenza: ogni aggregazione (`max`, `mean`, `temporal_topk_mean`) restituisce
lo stesso punteggio. Non c'e' SMOTE sul target: il loader di training non espone il
dominio del clip.

Il runner salva gli score continui di training come `train_anomaly_score_*.csv` e
gli score continui di test come `anomaly_score_*.csv`; non scrive decisioni
binarie per default. Dopo aver scelto il percentile, rigenera le decisioni senza
riaddestrare. Per il protocollo usato qui, la soglia viene calcolata direttamente
sugli score del test, separatamente per metodo e macchina:

```bash
python paper_benchmark/recompute_dcase2026_decisions.py \
  --percentile 95 \
  --methods cfa dinomaly stfpm
```

`--write-decisions --decision-percentile 99` può essere usato per generare le
decisioni direttamente durante il runner. Questa è una soglia calibrata sul test
e produce metriche binarie ottimistiche; AUC e pAUC continuano invece a usare gli
score continui.

### DCASE 2026 self-supervised lontano -> vicino

La pipeline self-supervised è separata dal runner DCASE standard. Pre-addestra
un encoder per macchina usando soltanto i file normali source del canale 1
(microfono lontano), quindi addestra il detector sui file normali source del
canale 0 (microfono vicino):

```bash
CUDA_VISIBLE_DEVICES=2 uv run python paper_benchmark/run_dcase2026_task2_ssl.py \
  --config paper_benchmark/config.dcase2026_ssl.example.json \
  --methods padim patchcore stfpm cfa dinomaly draem \
  --write-decisions
```

I checkpoint vengono riutilizzati tra PaDiM, PatchCore, STFPM e CFA. Dinomaly
riceve un checkpoint SSL separato perché usa un encoder ViT incompatibile con il
backbone audio. `--force-ssl-pretrain` forza il nuovo pretraining;
`--skip-ssl-pretrain` richiede checkpoint già esistenti; `--ssl-only` esegue
soltanto il pretraining. Per un controllo rapido usa `--debug --no-pretrained`.

La configurazione di esempio è
`paper_benchmark/config.dcase2026_ssl.example.json`. Il dataset può essere
impostato nel file oppure sovrascritto con `--dataset-path`. Gli output destinati
all'evaluator sono separati dalla pipeline standard e vengono scritti in
`dcase2026_task2_evaluator/dev_ssl`. Per valutarli:

```bash
cd dcase2026_task2_evaluator
uv run python dcase2026_task2_evaluator.py \
  --ground_truth_root dev_ssl
```

Il runner salva inoltre il riepilogo in
`results/paper_benchmark/dcase2026_task2_ssl_results.json`. La modalità di
default è sempre far-to-near; target e test non entrano nel pretraining SSL.

## Note pratiche

### Configurazione audio DCASE

Il frontend determina il sample rate: Cnn14 resta a 44100 Hz, HTSAT-base usa
48000 Hz. I WAV DCASE devono essere a 16000 Hz prima del risampling.
Nel runner standard si possono aggiungere alla sezione `dcase2026`:

```json
{
  "channel": 0,
  "train_domains": "source",
  "test_domain": "source",
  "run_name": "source_ch0_seed42"
}
```

I default restano canale 0, training e test su entrambi i domini. Il runner SSL
mantiene training source e i canali della sezione `dcase2026_ssl`.
`run_name` e' opzionale: separa gli export in
`dcase2026_task2_evaluator/dev/<run_name>` (oppure `dev_ssl/<run_name>` per SSL)
e i riepiloghi in `<output_dir>/dev/<run_name>` (oppure `dev_ssl/<run_name>`).
Scegliere un nome diverso per ciascun esperimento e per i debug; riutilizzare
lo stesso nome mantiene il comportamento di sovrascrittura dei file.
La configurazione e gli argomenti CLI vengono salvati insieme ai riepiloghi,
aggiornati dopo ogni detector completato.

Per rigenerare i binari di un run nominato, passare
`--teams-root dcase2026_task2_evaluator/dev/<run_name>/teams/moviad` a
`recompute_dcase2026_decisions.py`. Per SSL usare
`dev_ssl/<run_name>/teams/moviad_ssl`. Nell'evaluator passare come
`--ground_truth_root` la relativa directory `dev/<run_name>` o `dev_ssl/<run_name>`.

I nuovi state dict audio includono i moduli dell'estrattore; vecchi checkpoint
di detector privi di questi pesi non sono equivalenti e possono richiedere una
migrazione. Il formato dei checkpoint CLAP di ingresso resta invariato.
Il riuso SSL verifica anche seed, configurazione SSL e debug; in caso di
differenze usare un nuovo run o `--force-ssl-pretrain`.

- Se manca `clap_encoder.pth`, gli script con `pretrained: true` si fermano subito con un errore chiaro.
- Se il PC non ha GPU, usa `device: "cpu"` o `device: "auto"`, ma il run completo sara' lento.
- Se vuoi provare un solo modello, usa `--methods patchcore`, oppure `--methods padim`, ecc.
- Per dimezzare il training set e ridurre la memoria usata da PatchCore e PaDiM, aggiungi `--subset`. Puoi anche indicare una frazione, per esempio `--subset 0.25`. Il test set resta completo.
- PaDiM e PatchCore fanno il fit sempre a memoria limitata (un batch alla volta): PaDiM accumula media/covarianza in un solo passaggio, PatchCore riduce ogni batch con k-center-greedy e poi riduce il pool al `memory_bank_size`. `--streaming`/`padim_streaming` sono accettati ma non hanno più effetto. `force_cpu_coreset: true` esegue la selezione del coreset su CPU (utile se la GPU è piena).
- Se vuoi ridurre il benchmark, modifica in config `categories`, `snrs`, `seeds` o `background_categories`.
- PaDiM usa di default covarianza diagonale per ridurre il costo CPU/RAM. Per la
  covarianza completa imposta `padim_diag_cov: false`.
- Se i clip di test hanno una durata diversa da quelli di training, PaDiM
  ridimensiona l'embedding alla griglia di training emettendo un warning (invece di
  fallire): per punteggi esatti usa clip della stessa durata.
- I file `config.local.json`, CSV e report generati sono locali: di norma non serve committarli.
