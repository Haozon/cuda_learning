# nsys profile --stats=true --force-overwrite true -o baseline python baseline.py -size 512
nsys profile --stats=true --force-overwrite true -o gevm_int8 ./gemv -s 2048 -x 32 -y 8 -i 100 -b 8