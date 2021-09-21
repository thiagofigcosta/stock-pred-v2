#!/bin/bash

if [[ "$RUN_DEFAULT_EXP" = [tT][rR][uU][eE] ]]; then
    python main.py --dummy --clear-plots-models-and-datasets \
    echo -e "2018\n\n" | tee -a log.txt; \
    python main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --start-date 01/01/2018 --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup | tee -a log.txt; \
    echo -e "\n\n\n\n2015\n\n" | tee -a log.txt; \
    python main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --start-date 01/01/2015 --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup | tee -a log.txt; \
    echo -e "\n\n\n\nALL\n\n" | tee -a log.txt; \
    python main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup | tee -a log.txt \
    python main.py --dummy --restore-backups | tee -a log.txt; \
    echo -e "\n\n\nDONE\n" | tee -a log.txt
elif [[ "$RUN_IRACE_NAS" = [tT][rR][uU][eE] ]]; then
    (cd /code/irace ; /usr/local/lib/R/site-library/irace/bin/irace | tee ../log.txt)
    echo -e "\n\n\nDONE\n" | tee -a log.txt
elif [[ "$RUN_GENETIC_NAS" = [tT][rR][uU][eE] ]]; then
    python nas_genetic.py | tee log.txt
    echo -e "\n\n\nDONE\n" | tee -a log.txt
fi

if [[ "$COMPRESS_AFTER_EXP" = [tT][rR][uU][eE] ]]; then
    tar -zcvf /code/exp.tar.gz /code/datasets /code/saved_models /code/saved_plots /code/irace /code/log.txt
fi

tail -f /dev/null # to keep running
