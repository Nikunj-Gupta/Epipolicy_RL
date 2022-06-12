import json, requests, threading, time, uvicorn, socket, subprocess, resource
from fastapi import FastAPI
from urllib3.connection import HTTPConnection

from .epidemic import Epidemic
from .. import conf as conf
from ..conf import STANDALONE_HOST, STANDALONE_PORT

ULIMIT = 10000
WARMUP_TIME = 14
SLEEP_TIME = 1
RETRY_TIME = 0.5

# Solution to solve "Too many open files"
resource.setrlimit(resource.RLIMIT_NOFILE, (ULIMIT, ULIMIT))
# Mac only
# print("Current ulimit:", str(subprocess.check_output(["ulimit", "-n"]), 'utf-8'), end='')

warmup_scenario = r"""
{"status":"start","id":"1","name":"Warmup","session":{"features":{"start_date":"2021-01-01","end_date":"2021-12-31","data_source":"GPWv4","region":"UP","level":1,"year":2020},"interventions":[{"id":"1","name":"Vaccination","notes":"","control_params":[{"id":"1","name":"degree","description":"describe parameter 1 here...","default_value":"0"},{"id":"2","name":"max_capacity","description":"describe parameter 2 here...","default_value":"20000"},{"id":"3","name":"price_per_dose","description":"describe parameter 3 here...","default_value":"20"}],"effect":"def effect(cp, locales):\n    sim.move({'compartment':'S', 'locale':locales}, {'compartment':'V1', 'locale':locales}, cp['degree']*cp['max_capacity'])\n","cost":"def cost(cp, locales):\n    doses = cp['degree']*cp['max_capacity']\n    sim.add({\"intervention\":\"Vaccination\", 'locale':locales}, doses*cp['price_per_dose'])\n"},{"id":"2","name":"Mask","notes":"","control_params":[{"id":"1","name":"compliance","description":"describe parameter 1 here...","default_value":"0"},{"id":"2","name":"cost_per_day","description":"describe parameter 2 here...","default_value":"0.05"}],"effect":"def effect(cp, locales):\n    sim.apply({'parameter':'beta', 'locale':locales}, 1-cp['compliance']*0.7)\n","cost":"def cost(cp, locales):\n    complianceCount = sim.select({'compartment':'*', 'locale':locales})['Value'].sum() * cp['compliance']\n    sim.add({\"intervention\":\"Mask\"}, complianceCount*cp['cost_per_day'])\n"}],"costs":[{"id":"1","name":"Infectious_cost","notes":"Infectious people burden the hospital system","intervention":null,"control_params":[{"id":"1","name":"cost_per_day","description":"describe parameter 1 here...","default_value":"100"}],"func":"def cost(cp):\n    infCount = sim.select({'compartment':'{\"tag\":\"infectious\"}'})['Value'].sum()\n    sim.add({\"intervention\":\"Infectious_cost\"}, infCount*cp['cost_per_day'])"}],"locales":[{"population":1012394,"area":99324,"name":"UnitedProvinces","id":"UP","parent_id":""},{"population":202478.8,"area":44619,"name":"UnitedProvinces.Pastures","id":"UP.Pastures","parent_id":"UP"},{"population":313842.14,"area":24504,"name":"UnitedProvinces.Hills","id":"UP.Hills","parent_id":"UP"},{"population":496073.06,"area":30201,"name":"UnitedProvinces.Beaches","id":"UP.Beaches","parent_id":"UP"}],"model":{"name":"Two-dose vaccine SIR","compartments":[{"id":1,"name":"S","desc":"Susceptible","equation":"-(beta * I * S / N) + nu * R - (v1 * S)","tags":["susceptible"]},{"id":2,"name":"I","desc":"Infected","equation":"(beta * I * S / N) + (p1 * beta * I * V1 / N)\n - (gamma * I)","tags":["infectious"]},{"id":3,"name":"R","desc":"Recovered","equation":"gamma * I - (nu * R)","tags":[]},{"id":4,"name":"V1","desc":"One-dose Vaccinated","equation":"v1 * S - v2 * V1 - (p1 * beta * I * V1 / N)","tags":["susceptible"]},{"id":5,"name":"V2","desc":"Two-dose Vaccinated","equation":"v2 * V1","tags":["vaccinated"]}],"parameters":[{"id":1,"name":"beta","desc":"Transmission rate","default_value":"0.3","tags":["transmission"]},{"id":2,"name":"gamma","desc":"Recovery rate","default_value":"0.125","tags":[]},{"id":3,"name":"nu","desc":"Immunity-loss rate","default_value":"0.03","tags":[]},{"id":4,"name":"v1","desc":"One-dose vaccination rate","default_value":"0","tags":[]},{"id":5,"name":"v2","desc":"Two-dose vaccination rate","default_value":"0.0476","tags":[]},{"id":6,"name":"p1","desc":"Reduction from transmission after first dose","default_value":"0.5","tags":[]}]},"initial_info":{"name":"Warmup3","notes":"","initializers":[{"id":1,"locale_regex":"UnitedProvinces.*","group":"*","compartment":"I","value":"100"},{"id":2,"locale_regex":"UnitedProvinces.*","group":"*","compartment":"V1","value":0},{"id":3,"locale_regex":"UnitedProvinces.*","group":"*","compartment":"S","value":"2224426"}]},"groups":[{"name":"Seniors","description":"","locales":[{"name":"UnitedProvinces.Pastures","id":"UP.Pastures","population":0.063},{"name":"UnitedProvinces.Hills","id":"UP.Hills","population":0.063},{"name":"UnitedProvinces.Beaches","id":"UP.Beaches","population":0.063}],"properties":{"type":"GPWv4","gender":["m","f"],"age":[50,84]}},{"name":"Other","description":"","locales":[{"name":"UnitedProvinces.Pastures","id":"UP.Pastures","population":0.936},{"name":"UnitedProvinces.Hills","id":"UP.Hills","population":0.936},{"name":"UnitedProvinces.Beaches","id":"UP.Beaches","population":0.936}],"properties":{"type":"GPWv4","gender":["m","f"],"age":[0,49]}}],"group_specifications":[],"groups_locales_parameters":[{"id":0,"param":"beta","locale":"UnitedProvinces.Pastures","group":"Seniors","value":"0.3"},{"id":1,"param":"beta","locale":"UnitedProvinces.Hills","group":"Seniors","value":"0.3"},{"id":2,"param":"beta","locale":"UnitedProvinces.Beaches","group":"Seniors","value":"0.3"},{"id":3,"param":"gamma","locale":"UnitedProvinces.Pastures","group":"Seniors","value":"0.125"},{"id":4,"param":"gamma","locale":"UnitedProvinces.Hills","group":"Seniors","value":"0.125"},{"id":5,"param":"gamma","locale":"UnitedProvinces.Beaches","group":"Seniors","value":"0.125"},{"id":6,"param":"nu","locale":"UnitedProvinces.Pastures","group":"Seniors","value":"0.03"},{"id":7,"param":"nu","locale":"UnitedProvinces.Hills","group":"Seniors","value":"0.03"},{"id":8,"param":"nu","locale":"UnitedProvinces.Beaches","group":"Seniors","value":"0.03"},{"id":9,"param":"v1","locale":"UnitedProvinces.Pastures","group":"Seniors","value":"0"},{"id":10,"param":"v1","locale":"UnitedProvinces.Hills","group":"Seniors","value":"0"},{"id":11,"param":"v1","locale":"UnitedProvinces.Beaches","group":"Seniors","value":"0"},{"id":12,"param":"v2","locale":"UnitedProvinces.Pastures","group":"Seniors","value":"0.0476"},{"id":13,"param":"v2","locale":"UnitedProvinces.Hills","group":"Seniors","value":"0.0476"},{"id":14,"param":"v2","locale":"UnitedProvinces.Beaches","group":"Seniors","value":"0.0476"},{"id":15,"param":"p1","locale":"UnitedProvinces.Pastures","group":"Seniors","value":"0.5"},{"id":16,"param":"p1","locale":"UnitedProvinces.Hills","group":"Seniors","value":"0.5"},{"id":17,"param":"p1","locale":"UnitedProvinces.Beaches","group":"Seniors","value":"0.5"},{"id":18,"param":"beta","locale":"UnitedProvinces.Pastures","group":"Other","value":"0.3"},{"id":19,"param":"beta","locale":"UnitedProvinces.Hills","group":"Other","value":"0.3"},{"id":20,"param":"beta","locale":"UnitedProvinces.Beaches","group":"Other","value":"0.3"},{"id":21,"param":"gamma","locale":"UnitedProvinces.Pastures","group":"Other","value":"0.125"},{"id":22,"param":"gamma","locale":"UnitedProvinces.Hills","group":"Other","value":"0.125"},{"id":23,"param":"gamma","locale":"UnitedProvinces.Beaches","group":"Other","value":"0.125"},{"id":24,"param":"nu","locale":"UnitedProvinces.Pastures","group":"Other","value":"0.03"},{"id":25,"param":"nu","locale":"UnitedProvinces.Hills","group":"Other","value":"0.03"},{"id":26,"param":"nu","locale":"UnitedProvinces.Beaches","group":"Other","value":"0.03"},{"id":27,"param":"v1","locale":"UnitedProvinces.Pastures","group":"Other","value":"0"},{"id":28,"param":"v1","locale":"UnitedProvinces.Hills","group":"Other","value":"0"},{"id":29,"param":"v1","locale":"UnitedProvinces.Beaches","group":"Other","value":"0"},{"id":30,"param":"v2","locale":"UnitedProvinces.Pastures","group":"Other","value":"0.0476"},{"id":31,"param":"v2","locale":"UnitedProvinces.Hills","group":"Other","value":"0.0476"},{"id":32,"param":"v2","locale":"UnitedProvinces.Beaches","group":"Other","value":"0.0476"},{"id":33,"param":"p1","locale":"UnitedProvinces.Pastures","group":"Other","value":"0.5"},{"id":34,"param":"p1","locale":"UnitedProvinces.Hills","group":"Other","value":"0.5"},{"id":35,"param":"p1","locale":"UnitedProvinces.Beaches","group":"Other","value":"0.5"}],"facilities":[{"id":1,"name":"Household","description":"The household facility represents the pairwise connections between household members. Unlike schools and workplaces, everyone must be assigned to a household."},{"id":2,"name":"School","description":"The school facility represents all of the pairwise connections between people in schools, including both students and teachers. The current methods in SynthPops treat student and worker status as mutually exclusive."},{"id":3,"name":"Workplace","description":"The workplace facility represents all of the pairwise connections between people in workplaces, except for teachers working in schools. After some workers are assigned to the school contact layer as teachers, all remaining workers are assigned to workplaces. Workplaces are special in that there is little/no age structure so workers of all ages may be present in every workplace."},{"id":4,"name":"Community","description":"The community facility reflects the nature of contacts in shared public spaces like parks and recreational spaces, shopping centres, community centres, and public transportation. All links between individuals are considered undirected to reflect the ability of either individual in the pair to infect each other."}],"facilities_interactions":[{"locales":"UnitedProvinces.*","facilities":[[[1,0],[0,0]],[[0,0],[0,0]],[[1,0],[0,0]],[[1,0],[0,0]]]}],"facilities_timespent":[{"locales":"UnitedProvinces.*","matrix":[[0.53,0],[0,0],[0.19,0],[0.27,0]]}],"schedules":[{"id":"1","name":"Vaccination","notes":"describe schedule for Vaccination...","detail":[{"id":2,"start_date":"2021-01-01","end_date":"2021-02-28","control_params":[{"name":"degree","value":"0"},{"name":"max_capacity","value":"20000"},{"name":"price_per_dose","value":"20"}],"locales":"UnitedProvinces.*","has_trigger":false,"condition":"def isActive():\n\t# code here, this function must return a boolean value!\n\treturn True\n\t","repeat_config":{"repeat_type":"none","end_type":"never","end_date":"2021-12-31","end_times":0},"repetition_of":null}]},{"id":"2","name":"Mask","notes":"describe schedule for Mask...","detail":[{"id":1,"start_date":"2021-01-01","end_date":"2021-12-31","control_params":[{"name":"compliance","value":"1"},{"name":"cost_per_day","value":"0.05"}],"locales":"UnitedProvinces.*","has_trigger":false,"condition":"def isActive():\n\t# code here, this function must return a boolean value!\n\treturn True\n\t","repeat_config":{"repeat_type":"none","end_type":"never","end_date":"2021-12-31","end_times":0},"repetition_of":null}]}],"references":[],"border":{"data":[{"src_locale":"UnitedProvinces.Pastures","dst_locale":"UnitedProvinces.Beaches","src_locale_id":"UP.Pastures","dst_locale_id":"UP.Beaches","group":"*","value":0.4},{"src_locale":"UnitedProvinces.Hills","dst_locale":"UnitedProvinces.Beaches","src_locale_id":"UP.Hills","dst_locale_id":"UP.Beaches","group":"*","value":0.6},{"src_locale":"UnitedProvinces.Beaches","dst_locale":"UnitedProvinces.Pastures","src_locale_id":"UP.Beaches","dst_locale_id":"UP.Pastures","group":"*","value":0.3},{"src_locale":"UnitedProvinces.Beaches","dst_locale":"UnitedProvinces.Hills","src_locale_id":"UP.Beaches","dst_locale_id":"UP.Hills","group":"*","value":0.7}],"specifications":[{"id":1,"src_locale":"UnitedProvinces.*","dst_locale":"UnitedProvinces.*","group":"*","value":0,"impedance":70,"mobility_source":"GPWv4"}]},"airport":{"data":[],"specifications":[]},"facility":{"data":[],"specifications":[]}}}
"""

# Current issue with requests.post: https://github.com/psf/requests/issues/4937
# Below solution does not work on Mac
# HTTPConnection.default_socket_options = (
#     HTTPConnection.default_socket_options + [
#         (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), #Enables the feature
#         # (socket.SOL_TCP, socket.TCP_KEEPIDLE, 45), #Overrides the time when the stack willl start sending KeppAlives after no data received on a Persistent Connection
#         (socket.SOL_TCP, socket.TCP_KEEPINTVL, 10), #Defines how often thoe KA will be sent between them
#         (socket.SOL_TCP, socket.TCP_KEEPCNT, 6) #How many attemps will your code try if the server goes down before droping the connection.
#     ]
# )

# Temporary solution to solve: https://github.com/psf/requests/issues/4937
def async_post(response):
    try:
        requests.post(
            'http://' + conf.MANAGER_HOST + ':' + str(conf.MANAGER_PORT) + '/requests/post_response', json=response)
    except Exception as e:
        time.sleep(RETRY_TIME)
        async_post(response)

class Connection:

    def warmup(self):
        st = time.time()
        json_input = json.loads(warmup_scenario)
        epi = Epidemic(json_input)
        epi.run(T=WARMUP_TIME)
        print("Finished warmup run in {}s".format(time.time() - st), flush=True)


class ManagerConnection(Connection):
    def __init__(self, debug=False):

        if not debug:
            self.warmup()

        while True:
            try:
                self.get_request()
            except Exception as e:
                print(e, flush=True)
            time.sleep(SLEEP_TIME)

    def get_request(self):
        response = requests.get(
            'http://' + conf.MANAGER_HOST + ':' + str(conf.MANAGER_PORT) + '/requests/get_new_request/')
        obj = json.loads(response.content)
        if len(obj['result']) > 0:
            json_input = obj['result'][0]
            try:
                epi = Epidemic(json_input, connection=self, config={"debug":True})
                epi.run()
            except Exception as e:
                print(e, flush=True)
                self.update(json_input["id"], "abort")

    def post(self, response):
        x = threading.Thread(target=async_post, args=(response,))
        x.start()
        return x

    def update(self, sim_id, status):
        req = {"id": sim_id, "status": status}
        while True:
            try:
                requests.post(
                    'http://' + conf.MANAGER_HOST + ':' + str(conf.MANAGER_PORT) + '/requests/update_status', json=req)
                break
            except Exception as e:
                print(e, flush=True)
                time.sleep(SLEEP_TIME)


class SyncEpidemicRunner:

    def __init__(self, json_input):
        self.res = []
        self.lock = threading.Lock()
        self.response = None
        self.epidemic = Epidemic(json_input, connection=self, config={"debug":True})

    def run(self):
        print('Starting simulation.', end='', flush=True)
        self.epidemic.run()
        return self.res

    def post(self, response):
        print('.', response, end='', flush=True)
        self.res.append(response)

    def update(self, sim_id, status):
        print('Simulation {} running: {} '.format(sim_id, status), flush=True)


app = FastAPI()


@app.post("/")
def root(req: dict):
    runner = SyncEpidemicRunner(req)
    return runner.run()


class StandaloneConnection(Connection):
    def __init__(self, debug=False):
        if not debug:
            self.warmup()

        uvicorn.run(app, host=STANDALONE_HOST, port=STANDALONE_PORT)
