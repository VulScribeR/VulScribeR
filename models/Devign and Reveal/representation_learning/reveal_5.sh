echo "Starting reveal" 2>&1 | tee reveal_results.txt

python reveal.py 2>&1 | tee -a reveal_results.txt
echo "DONE 1" 2>&1 | tee -a reveal_results.txt
python reveal.py 2>&1 | tee -a reveal_results.txt
echo "DONE 2" | tee -a reveal_results.txt
python reveal.py 2>&1 | tee -a reveal_results.txt
echo "DONE 3" 2>&1 | tee -a reveal_results.txt
python reveal.py 2>&1 | tee -a reveal_results.txt
echo "DONE 4" 2>&1 | tee -a reveal_results.txt
python reveal.py 2>&1 | tee -a reveal_results.txt
echo "DONE 5" 2>&1 | tee -a reveal_results.txt
