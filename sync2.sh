rsync -rzvhh rgerum@cedar.computecanada.ca:~/scratch/dimension_regularisation/logs/* cedar_logs --exclude='*alpha2.csv'
#find . -depth -type d -empty -delete
#ssh rgerum@cedar.computecanada.ca find /home/rgerum/scratch/dimension_regularisation/logs -depth -type d -empty -not -path /home/rgerum/scratch/dimension_regularisation/logs -delete
