#!/bin/bash

if [[ "$RUN_DEFAULT_EXP" = [tT][rR][uU][eE] ]]; then
    python main.py --dummy --clear-plots-models-and-datasets \
    echo -e "2018\n\n" >> log.txt; \
    python main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --start-date 01/01/2018 --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup >> log.txt; \
    echo -e "\n\n\n\n2015\n\n" >> log.txt; \
    python main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --start-date 01/01/2015 --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup >> log.txt; \
    echo -e "\n\n\n\nALL\n\n" >> log.txt; \
    python main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup >> log.txt \
    python main.py --dummy --restore-backups >> log.txt; \
    echo -e "\n\n\nDONE\n" >> log.txt
fi

tail -f /dev/null # to keep running