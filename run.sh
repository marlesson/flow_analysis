#bin/bash

# python parse_lk.py -f data/sample.flv -s 10 -c 1000 -out experimentos/output/sample &
# python parse_lk.py -f data/scene03_x1_view1.mpg -s 3 -c 1000 -out experimentos/output/scene03_x1_view1 &
# python parse_lk.py -f data/scene06_x1.avi -s 3 -c 1000 -out experimentos/output/scene06_x1 &
# python parse_lk.py -f data/scene02_x1_view1.avi -s 3 -c 1000 -out experimentos/output/scene02_x1_view1 &
# python parse_lk.py -f data/scene01_x1_view1_1.avi -s 3 -c 1000 -out experimentos/output/scene01_x1_view1_1 &
# python parse_lk.py -f data/scene07_x4_view1.avi -s 3 -c 1000 -out experimentos/output/scene07_x4_view1 &
# python parse_lk.py -f data/scene04_x1_view1.avi -s 3 -c 1000 -out experimentos/output/scene04_x1_view1 &
# python parse_lk.py -f data/scene05_view1.avi -s 3 -c 1000 -out experimentos/output/scene05_view1 &

python parse_arrows.py -f experimentos/output/sample.csv -out experimentos/output &
python parse_arrows.py -f experimentos/output/scene03_x1_view1.csv -out experimentos/output &
python parse_arrows.py -f experimentos/output/scene06_x1.csv -out experimentos/output &
python parse_arrows.py -f experimentos/output/scene02_x1_view1.csv -out experimentos/output &
python parse_arrows.py -f experimentos/output/scene01_x1_view1_1.csv -out experimentos/output &
python parse_arrows.py -f experimentos/output/scene07_x4_view1.csv -out experimentos/output &
python parse_arrows.py -f experimentos/output/scene04_x1_view1.csv -out experimentos/output &
python parse_arrows.py -f experimentos/output/scene05_view1.csv -out experimentos/output &
