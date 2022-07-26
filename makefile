all: 
	clear 
	python3 runner.py  --exp ppo_rs_1056 --config configs/ppo.yaml --scenario jsons/SIRV_A.json --algo ppo 
	python3 runner.py  --exp ppo_rs_3445 --config configs/ppo.yaml --scenario jsons/SIRV_A.json --algo ppo 
	python3 runner.py  --exp ppo_rs_5952 --config configs/ppo.yaml --scenario jsons/SIRV_A.json --algo ppo 
	python3 runner.py  --exp ppo_rs_9774 --config configs/ppo.yaml --scenario jsons/SIRV_A.json --algo ppo 

sac: 
	clear 
	# python3 runner.py  --exp sac_rs_1817 --config configs/sac.yaml --scenario jsons/SIRV_A.json --algo sac 
	# python3 runner.py  --exp sac_rs_2178 --config configs/sac.yaml --scenario jsons/SIRV_A.json --algo sac 
	# python3 runner.py  --exp sac_rs_3535 --config configs/sac.yaml --scenario jsons/SIRV_A.json --algo sac 
	# python3 runner.py  --exp sac_rs_9455 --config configs/sac.yaml --scenario jsons/SIRV_A.json --algo sac 
	python3 runner.py  --exp sac_rs_10 --config configs/sac.yaml --scenario jsons/SIRV_A.json --algo sac 
	


build_runs: 
	clear 
	python3 run_utils.py --algo sac 
	python3 run_utils.py --algo ppo 

config: 
	clear 
	python3 configs/config_rs_utils.py --algo sac 
	python3 configs/config_rs_utils.py --algo ppo 

run: 
	# python3 runner.py  --exp exp_7 --config configs/config.yaml --scenario jsons/SIRV_A.json --algo ppo 
	# python3 runner.py  --exp exp_7 --config configs/config.yaml --scenario jsons/SIRV_A.json --algo sac 

req: 
	pip3 install -r requirements.txt 

SCENARIOS = SIRV_A.json SIRV_B.json SIR_A.json SIR_B.json COVID_A.json COVID_B.json COVID_C.json 
test: 
	# $(foreach var,$(SCENARIOS),python3 runner.py  --exp ppo_rs_18 --config configs/ppo.yaml --scenario jsons/$(var) --algo ppo;)
	$(foreach var,$(SCENARIOS),python3 runner.py  --exp sac_rs_4 --config configs/sac.yaml --scenario jsons/$(var) --algo sac;)
