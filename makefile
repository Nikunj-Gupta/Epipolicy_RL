all: 
	clear 
	time python3 runner.py  --exp exp_7 --config configs/config.yaml --scenario jsons/SIRV_A.json

build_runs: 
	python3 run_utils.py 

config: 
	python3 configs/config_utils.py 

run: 
	python3 runner.py --exp exp_name --config configs/sample.yaml --scenario jsons/SIRV_A.json

req: 
	pip3 install -r requirements.txt 