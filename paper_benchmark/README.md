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

Per ottenere performance sensate devi usare il backbone pre-addestrato. Metti il checkpoint qui:

```text
src/moviad/weights/clap_encoder.pth
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

## Note pratiche

- Se manca `clap_encoder.pth`, gli script con `pretrained: true` si fermano subito con un errore chiaro.
- Se il PC non ha GPU, usa `device: "cpu"` o `device: "auto"`, ma il run completo sara' lento.
- Se vuoi provare un solo modello, usa `--methods patchcore`, oppure `--methods padim`, ecc.
- Se vuoi ridurre il benchmark, modifica in config `categories`, `snrs`, `seeds` o `background_categories`.
- I file `config.local.json`, CSV e report generati sono locali: di norma non serve committarli.
