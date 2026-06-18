#!/usr/bin/env bash
# One-shot retry for 5 corrupt-gzip failures from lee2022 retry pass.
# Deletes any partial files, re-downloads with 1 connection, runs fastp + kraken2.
# Run from git root: nohup bash cancer_project/scripts/retry_5_failed.sh > cancer_project/logs/lee2022_retry2.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

REPORT_DIR="results/kraken_reports/lee2022"
FASTP_DIR="results/fastp/lee2022"
RAW_DIR="raw_data/lee2022"
TRIM_DIR="results/trimmed/lee2022"
KO_DIR="results/kraken_outputs/lee2022"
THREADS="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 8)"

mkdir -p "$REPORT_DIR" "$FASTP_DIR" "$RAW_DIR" "$TRIM_DIR" "$KO_DIR" logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(ts)  $1"; }
safe_rm() { for f in "$@"; do [[ -f "$f" ]] && rm -f "$f"; done; }

log "=== Lee 2022 RETRY-2 (5 failed samples) ==="

if [[ ! -f "hash.k2d" ]]; then
    log "ERROR: hash.k2d not found in ${PROJECT_ROOT}"; exit 1
fi


# 5 failed samples: "run url1 url2"
SAMPLES=(
  "ERR6275667 https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR627/007/ERR6275667/ERR6275667_1.fastq.gz https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR627/007/ERR6275667/ERR6275667_2.fastq.gz"
  "ERR6275672 https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR627/002/ERR6275672/ERR6275672_1.fastq.gz https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR627/002/ERR6275672/ERR6275672_2.fastq.gz"
  "ERR6275675 https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR627/005/ERR6275675/ERR6275675_1.fastq.gz https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR627/005/ERR6275675/ERR6275675_2.fastq.gz"
  "ERR6275676 https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR627/006/ERR6275676/ERR6275676_1.fastq.gz https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR627/006/ERR6275676/ERR6275676_2.fastq.gz"
  "ERR6279623 https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR627/003/ERR6279623/ERR6279623_1.fastq.gz https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR627/003/ERR6279623/ERR6279623_2.fastq.gz"
)
SUCCESS=0; FAILED=0; IDX=0; TOTAL=${#SAMPLES[@]}

for entry in "${SAMPLES[@]}"; do
    run=$(echo "$entry" | awk '{print $1}')
    url1=$(echo "$entry" | awk '{print $2}')
    url2=$(echo "$entry" | awk '{print $3}')
    IDX=$((IDX + 1))
    REPORT="${REPORT_DIR}/${run}_report.txt"

    if [[ -f "$REPORT" ]]; then
        log "[$run] ($IDX/$TOTAL) SKIP — already done"
        SUCCESS=$((SUCCESS + 1)); continue
    fi

    log "[$run] ($IDX/$TOTAL) START"

    R1="${RAW_DIR}/${run}_1.fastq.gz"
    R2="${RAW_DIR}/${run}_2.fastq.gz"
    T1="${TRIM_DIR}/${run}_1.trimmed.fastq.gz"
    T2="${TRIM_DIR}/${run}_2.trimmed.fastq.gz"
    KO="${KO_DIR}/${run}_output.txt"

    # Wipe any corrupt partial files
    safe_rm "$R1" "$R2" "$T1" "$T2" "$KO"

    # Download R1
    log "  [$run] download R1"
    if ! aria2c -x1 -s1 --continue=false --max-tries=5 --retry-wait=30 \
         -d "$RAW_DIR" -o "${run}_1.fastq.gz" "$url1" 2>&1 | \
         grep -E "Download complete|ERROR|FAILED"; then
        log "[$run] FAILED: R1 download error"; FAILED=$((FAILED + 1)); continue
    fi

    # Verify R1 gzip integrity
    if ! gzip -t "$R1" 2>/dev/null; then
        log "[$run] FAILED: R1 corrupt gzip after download"; safe_rm "$R1"
        FAILED=$((FAILED + 1)); continue
    fi

    # Download R2
    log "  [$run] download R2"
    if ! aria2c -x1 -s1 --continue=false --max-tries=5 --retry-wait=30 \
         -d "$RAW_DIR" -o "${run}_2.fastq.gz" "$url2" 2>&1 | \
         grep -E "Download complete|ERROR|FAILED"; then
        log "[$run] FAILED: R2 download error"; safe_rm "$R1"; FAILED=$((FAILED + 1)); continue
    fi

    # Verify R2 gzip integrity
    if ! gzip -t "$R2" 2>/dev/null; then
        log "[$run] FAILED: R2 corrupt gzip after download"; safe_rm "$R1" "$R2"
        FAILED=$((FAILED + 1)); continue
    fi

    log "  [$run] download OK"

    # fastp
    log "  [$run] fastp START"
    JSON="${FASTP_DIR}/${run}.json"
    HTML="${FASTP_DIR}/${run}.html"
    if ! fastp -i "$R1" -I "$R2" -o "$T1" -O "$T2" \
         -j "$JSON" -h "$HTML" \
         --thread "$THREADS" --detect_adapter_for_pe \
         --cut_front --cut_tail --cut_mean_quality 20 \
         --length_required 50 2>&1 | tail -3; then
        log "[$run] FAILED: fastp error"; safe_rm "$R1" "$R2" "$T1" "$T2"
        FAILED=$((FAILED + 1)); continue
    fi
    safe_rm "$R1" "$R2"
    log "  [$run] fastp OK"

    # Kraken2
    log "  [$run] kraken2 START"
    if ! kraken2 --db . --paired --threads "$THREADS" \
         --report "$REPORT" --output "$KO" \
         "$T1" "$T2" 2>&1 | tail -3; then
        log "[$run] FAILED: kraken2 error"; safe_rm "$T1" "$T2" "$REPORT"
        FAILED=$((FAILED + 1)); continue
    fi
    safe_rm "$T1" "$T2" "$KO"

    log "[$run] DONE — report saved"
    SUCCESS=$((SUCCESS + 1))
done

log "=== RETRY-2 COMPLETE === Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED"
