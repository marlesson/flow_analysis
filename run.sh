#bin/bash

python parse_lk.py -f data/scene01_x1_view1_1.avi -s 3 -c 1000 -out output/scene01_x1_view1_1
python parse_lk.py -f data/scene01_x4_view1.avi -s 3 -c 1000 -out output/scene01_x4_view1
python parse_lk.py -f data/scene02_x1_view1.avi -s 3 -c 1000 -out output/scene02_x1_view1
python parse_lk.py -f data/scene07_x4_view1.avi -s 3 -c 1000 -out output/scene07_x4_view1
python parse_lk.py -f data/sample.flv -s 10 -c 1000 -out output/sample

python parse_lk.py -f data/scene03_x1_view2.mpg -s 3 -c 1000 -out output/scene03_x1_view2
python parse_lk.py -f data/scene03_x1_view1.mpg -s 3 -c 1000 -out output/scene03_x1_view1
python parse_lk.py -f data/scene02_x1_view2.avi -s 3 -c 1000 -out output/scene02_x1_view2

python parse_lk.py -f data/scene05_view1.avi -s 3 -c 1000 -out output/scene05_view1
python parse_lk.py -f data/scene05_view2.avi -s 3 -c 1000 -out output/scene05_view2

