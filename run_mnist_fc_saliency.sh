echo "CEX=$1"
python check_saliency_map_mnist.py $1 > encoding_stdout_$1 2>encoding_stderr_$1
time ~/workspace/Marabou/build/bin/Marabou --input-query=finalQuery_smallConv --snc --num-workers=30 --export-assignment --timeout=1200 > solving_std_$1 2>solving_stderr_$1
mv grad_bounds.txt grad_bounds_$1.txt
mv assignment.txt assignment$1.txt
mv finalQuery_smallConv InputQueryCEX$1