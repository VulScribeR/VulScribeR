echo "STARTING" 2>&1 | tee devign_results.txt
python devign_demo.py --dataset demo --input_dir ./demo/ --node_tag node_features --graph_tag graph --label_tag targets | tee -a devign_results.txt
echo "READY for REVEAL - done 1" 2>&1 | tee -a devign_results.txt
python devign_demo.py --dataset demo --input_dir ./demo/ --node_tag node_features --graph_tag graph --label_tag targets | tee -a devign_results.txt
echo "DONE 2" 2>&1 | tee -a devign_results.txt
python devign_demo.py --dataset demo --input_dir ./demo/ --node_tag node_features --graph_tag graph --label_tag targets | tee -a devign_results.txt
echo "DONE 3" 2>&1 | tee -a devign_results.txt
python devign_demo.py --dataset demo --input_dir ./demo/ --node_tag node_features --graph_tag graph --label_tag targets | tee -a devign_results.txt
echo "DONE 4" 2>&1 | tee -a devign_results.txt
python devign_demo.py --dataset demo --input_dir ./demo/ --node_tag node_features --graph_tag graph --label_tag targets | tee -a devign_results.txt
echo "DONE 5" 2>&1 | tee -a devign_results.txt
