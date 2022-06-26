all: 
	clear 
	python3 runner.py  --exp exp_10 --config configs/config_new.yaml --scenario jsons/SIRV_A.json --algo sac 

build_runs: 
	python3 run_utils.py 

config: 
	clear 
	python3 configs/config_utils.py 

run: 
	# python3 runner.py  --exp exp_7 --config configs/config.yaml --scenario jsons/SIRV_A.json --algo ppo 
	# python3 runner.py  --exp exp_7 --config configs/config.yaml --scenario jsons/SIRV_A.json --algo sac 

req: 
	pip3 install -r requirements.txt 