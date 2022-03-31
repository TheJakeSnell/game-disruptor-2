import asyncio
import base64
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.pool import ThreadPool
import matplotlib
import websockets.exceptions
import websockets.legacy
import argparse

matplotlib.use('Agg')
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from flask import Flask
from flask import render_template
from flask_assets import Environment

import webview
from threading import Lock, Thread
import os, sys
from os import listdir
from os.path import isfile, join
import subprocess
from subprocess import Popen, PIPE, CalledProcessError
import time, random
from deap import creator, base, tools, algorithms
import copy
import traceback
import numpy as np
import io
import socket
from websockets.server import serve
from websockets.client import connect
import platform

base_dir = '.'
if hasattr(sys, '_MEIPASS'): # or, untested: if getattr(sys, 'frozen', False):
    base_dir = os.path.join(sys._MEIPASS)

mutex = Lock()

nodeInfo = {}
remainingTasks = []
hosting_ip = ""
hostTime = 0
appData = {}
appDataBackup = {}
hof = []
fitness_graph = ""
pareto_graph = ""
pareto_results = {}
selected_hof = 0
selected_case = 0
paramCount = 0
scrollPosY = 0

appOpen = False
runonce = False
experiment_name = ""
runningEvolution = False
startProcessing = False
hosting = False

app = Flask(__name__,
        static_folder=os.path.join(base_dir, 'static'),
        template_folder=os.path.join(base_dir, 'templates'))



# Bundling src/main.css files into dist/main.css'
#css = Bundle("src/main.css", output="dist/main.css", filters="postcss")
assets = Environment(app)
assets.register("main_css", 'dist/main.css')
#css.build()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/landing')
def landing():
    config_files = [join("projects", f, "config.json") for f in listdir("projects") if not isfile(join("projects", f))]
    projects = []

    for path in config_files:
        with open(path) as json_file:
            j = json.load(json_file)
            j['folder_path'] = os.path.abspath(os.path.dirname(path))
            print(j['folder_path'])
            projects.append(j)

    return render_template('landing.html', projects=projects)

@app.route('/show_project')
def show_project():
    return render_template('show_project.html', title=appData['name'], appData=appData, hosting_ip=hosting_ip, hosting=hosting, selected_case=selected_case)

@app.route('/new_project')
def new_project():
    print(appData)
    return render_template('new_project.html', title=appData['name'], appData=appData, scrollPosY=scrollPosY)

@app.route('/edit_project')
def edit_project():
    print(appData)
    return render_template('edit_project.html', title=appData['name'], appData=appData, scrollPosY=scrollPosY)


@app.route('/run_evolution')
def run_evolution():
    return render_template('run_evolution.html', title=appData['name'], appData=appData)

@app.route('/connect')
def connect_page():
    return render_template('connect.html', title=appData['name'], appData=appData, host_ip=hosting_ip)

@app.route('/show_evolution')
def show_evolution():
    print(pareto_results['results'])
    return render_template('show_evolution.html', title=appData['name'], appData=appData, fitness_graph=fitness_graph, pareto_graph=pareto_graph, pareto_results=pareto_results, selected_hof=selected_hof)

def edit_existing_project():
    global appDataBackup
    appDataBackup = copy.deepcopy(appData)

    webview.windows[0].load_url(webview.windows[0].get_current_url().replace("show_project", "edit_project"))

def cancel_edit():
    global appData
    appData = copy.deepcopy(appDataBackup)

    webview.windows[0].load_url(webview.windows[0].get_current_url().replace("edit_project", "show_project"))

def save_project():
    path = appData['folder_path'] + "/config.json"
    with open(path, "w") as outfile:
        outfile.write(json.dumps(appData))

    webview.windows[0].load_url(webview.windows[0].get_current_url().replace("edit_project", "show_project"))

def show_simulation():
    config = copy.deepcopy(appData)

    game_vars = {}
    for key in config['game_variables'].keys():
        game_vars[key] = config['game_variables'][key]['value']

    config['game_variables'] = game_vars

    asyncio.run(call_simulator(display=True, config=config))

def host():
    global hosting
    global hosting_ip
    global nodeInfo
    global remainingTasks

    if hosting:
        return

    hosting = True

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
    hosting_ip = ip_address

    url = webview.windows[0].get_current_url()
    webview.windows[0].load_url(webview.windows[0].get_current_url().replace("show_project", "connect"))
    webview.windows[0].load_url(url)

    async def handler(websocket):
        while True:
            await asyncio.sleep(0.1)
            try:
                if str(websocket.id) in nodeInfo.keys():
                    node_info = nodeInfo[str(websocket.id)]
                    if startProcessing and len(node_info['tasks']) > len(node_info['results']):
                        # Send tasks to be processed
                        print(node_info)
                        print("Sending "+str(len(node_info['tasks']))+" tasks to: "+str(websocket.remote_address))
                        await websocket.send(json.dumps(node_info))

                        # Read response
                        resp = json.loads(await websocket.recv())

                        nodeInfo[str(websocket.id)]['results'] = resp['results']
                        if nodeInfo[str(websocket.id)]['avgTime'] > 0:
                            nodeInfo[str(websocket.id)]['avgTime'] = (nodeInfo[str(websocket.id)]['avgTime'] + resp['avgTime']) / 2
                        else:
                            nodeInfo[str(websocket.id)]['avgTime'] = resp['avgTime']

                else:
                    while True:
                        if not startProcessing:
                            break

                        await asyncio.sleep(0.1)

                    # Handle new connection
                    print("Node Connected!")
                    node_info = json.loads(await websocket.recv())

                    nodeInfo[str(websocket.id)] = node_info
                    print(str(nodeInfo))

                    #await websocket.send("Wait for work!")

            except websockets.exceptions.ConnectionClosed as ex:
                print("Node Disconnected!")
                print(str(ex))

                for task in nodeInfo[str(websocket.id)]['tasks']:
                    remainingTasks.append(task)


                del nodeInfo[str(websocket.id)]

                print(str(nodeInfo))
                break

    async def mainHost():
        async with serve(handler, ip_address, 13294, ping_timeout=180, close_timeout=60, ping_interval=180, max_size=None):
            await asyncio.Future()  # run forever

    asyncio.run(mainHost())

def connectToHost(ip_address):
    global hosting_ip
    hosting_ip = ip_address

    webview.windows[0].load_url(webview.windows[0].get_current_url().replace("show_project", "connect"))
    webview.windows[0].evaluate_js("document.getElementById('status_text').innerHTML='Connecting to: "+ str(ip_address) +"'")
    print("Attempting to connect to: "+str(ip_address))

    async def communicate(uri):
        async for websocket in websockets.connect(uri):
            try:
                print("Connection Successful!")
                webview.windows[0].evaluate_js(
                    "document.getElementById('status_text').innerHTML='Connected Successfully! Waiting for work from host...'")

                node_info = {
                    "cores": multiprocessing.cpu_count(),
                    "tasks": [],
                    "results": [],
                    "avgTime": 0.0
                }
                await websocket.send(json.dumps(node_info))

                # process any requests
                while True:
                    await asyncio.sleep(0.1)

                    node_info = json.loads(await websocket.recv())
                    print(node_info)

                    webview.windows[0].evaluate_js(
                        "document.getElementById('status_text').innerHTML='Processing "+str(len(node_info['tasks']))+" tasks...'")

                    results = []

                    pool = ThreadPool(processes=len(node_info['tasks']))
                    async_results = [pool.apply_async(call_simulator, (False, config)) for config in
                                     node_info['tasks']]

                    totalTime = 0
                    for idx, async_result in enumerate(async_results):
                        response, elapsedTime = async_result.get()
                        result = json.loads(response)

                        totalTime += elapsedTime

                        results.append(result)

                    avgTime = totalTime / len(async_results)

                    msg = json.dumps({"results": results, "avgTime": avgTime})
                    print(msg)

                    webview.windows[0].evaluate_js(
                        "document.getElementById('status_text').innerHTML='Processing Completed! Waiting for work from host...'")

                    await websocket.send(msg)
            except websockets.ConnectionClosed:
                print("Connection closed! Reconnecting...")
                continue

    asyncio.run(communicate("ws://"+str(ip_address)+":13294"))



def call_simulator(display, config):
    args = appData['simulator_command'].split(' ')
    if(display):
        args.append("display")

    response = ''

    startTime = time.time()

    try:
        os = platform.platform()
        if "mac" in os:
            runWithShell = False
        else:
            runWithShell = True

        with Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True, cwd=appData['simulator_path'], shell=runWithShell) as p:
            config['scenario'] = config['scenarios'][0]
            del config['scenarios']

            out, err = p.communicate(input=json.dumps(config))

            lines = out.split("\n")
            response = lines[len(lines)-2]

            if len(err) > 0:
                print(err)
    except Exception as e:
        print("Failed to call simulator! ")
        print(e)

    endTime = time.time()
    elapsedTime = endTime - startTime

    return response, elapsedTime

# New mutation function - randomally mutate the game variables checking the upper limit against the configured constraints
def mutateUniform(individual, indpb, limits):
    for i in range(len(individual)):
        if random.random() < indpb:
            #rand = individual[i] + (random.gauss(0, 0.2) * (limits[i]))
            if 'integer' in limits[i]['type']:
                rand = random.randint(limits[i]['min'], limits[i]['max'])
                #rand = int(individual[i] + (random.gauss(0, limits[i]['max'] - limits[i]['min'])))
            else:
                rand = random.uniform(limits[i]['min'], limits[i]['max'])
                #rand = individual[i] + (random.gauss(0, limits[i]['max'] - limits[i]['min']))

            individual[i] = max(limits[i]['min'], min(rand, limits[i]['max']))

    return individual,

# def randUniform(low, high, limits):
#

async def parralell_simulation(coros):
    return await asyncio.gather(*coros)


def view_experiment():
    global fitness_graph
    global pareto_graph
    global pareto_results

    path = os.path.abspath(appData['simulator_path']+"/experiments")
    result = webview.windows[0].create_file_dialog(dialog_type=webview.FOLDER_DIALOG,directory=path, allow_multiple=False)[0]

    with open(join(result, "pareto.json")) as json_file:
        pareto_results = json.load(json_file)

    with open(join(result, "fitness.png"), "rb") as image_file:
        fitness_graph = 'data:image/png;base64,'+base64.b64encode(image_file.read()).decode()

    with open(join(result, "pareto.png"), "rb") as image_file:
        pareto_graph = 'data:image/png;base64,'+base64.b64encode(image_file.read()).decode()

    webview.windows[0].load_url(webview.windows[0].get_current_url().replace("show_project", "show_evolution"))

    webview.windows[0].evaluate_js("document.getElementById('pareto-list').innerHTML=''")

    for idx, h in enumerate(pareto_results['results']):
        objective_keys = list(h['objectives'].keys())
        option_text = objective_keys[0] + ":</i> " + str(h['objectives'][objective_keys[0]]) + ",     " + objective_keys[1] + ": " + str(h['objectives'][objective_keys[1]])
        js_script = "document.getElementById('pareto-list').innerHTML += '<option value=\""+ str(idx) +"\" >"+ option_text +"</option>'"
        webview.windows[0].evaluate_js(js_script)

    webview.windows[0].evaluate_js("document.getElementById('pareto-list').value='0'")

    pareto_selected(0)

def remove_case():
    global selected_case
    global appData

    del appData['cases'][selected_case]

    if selected_case > 0:
        selected_case -= 1

    case_selected(selected_case)

def case_selected(idx):
    global selected_case
    global appData
    selected_case = int(idx)

    if len(appData['cases']) > 0:
        for key in appData['cases'][selected_case].keys():
            appData['game_variables'][key]['value'] = appData['cases'][selected_case][key]

    path = appData['folder_path'] + "/config.json"
    with open(path, "w") as outfile:
        outfile.write(json.dumps(appData))

    webview.windows[0].load_url(webview.windows[0].get_current_url())

def pareto_selected(idx):
    # Update statistics UI
    selected_stats = pareto_results['results'][int(idx)]['scenario_statistics']
    selected_distributions = pareto_results['results'][int(idx)]['scenario_statistics_distribution']

    webview.windows[0].evaluate_js("document.getElementById('statistics').innerHTML=''")

    for key in selected_stats.keys():
        webview.windows[0].evaluate_js(
            "document.getElementById('statistics').innerHTML += '<div class=\"pr-5 pb-2 text-sm text-gray-400\"><i>" + key + ":</i> " + str(
                selected_stats[key]) + "</div>'")

    # Update Game Variables UI
    webview.windows[0].evaluate_js("document.getElementById('variables').innerHTML=''")

    for key in pareto_results['results'][int(idx)]['game_variables'].keys():
        webview.windows[0].evaluate_js(
            "document.getElementById('variables').innerHTML += '<div class=\"pr-5 pb-2 text-sm text-gray-400\"><i>" +
            key + ":</i> " + str(pareto_results['results'][int(idx)]['game_variables'][key]) + "</div>'")

    # Update Statistics Distributions
    webview.windows[0].evaluate_js("document.getElementById('distributions').innerHTML=''")

    for key in selected_distributions.keys():
        webview.windows[0].evaluate_js(
            "document.getElementById('distributions').innerHTML += '<div class=\"pr-5 pb-2 text-sm text-gray-400\"><i>" + key + ":</i></div>'")

        fig, ax = plt.subplots()

        ax.hist(selected_distributions[key], color='blue', edgecolor='black', bins=20)
        ax.set(xlabel=key, ylabel="Frequency", title=key+' Distribution: ')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        #plt.legend()

        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

        webview.windows[0].evaluate_js(
            "document.getElementById('distributions').innerHTML += '<img class=\"w-1/2\" src=\"data:image/jpeg;base64, " +str(
            my_base64_jpgData) + "\">'")

        plt.close(fig)


def stop_evolving():
    global runningEvolution

    runningEvolution = False

    webview.windows[0].load_url(webview.windows[0].get_current_url().replace("run_evolution", "show_project"))

def hof_selected(idx):
    # Update statistics UI
    selected_stats = hof[int(idx)].stats
    selected_distributions = hof[int(idx)].distributions

    webview.windows[0].evaluate_js("document.getElementById('statistics').innerHTML=''")

    for key in selected_stats.keys():
        webview.windows[0].evaluate_js(
            "document.getElementById('statistics').innerHTML += '<div class=\"pr-5 pb-2 text-sm text-gray-400\"><i>" + key + ":</i> " + str(
                selected_stats[key]) + "</div>'")

    # Update Game Variables UI
    webview.windows[0].evaluate_js("document.getElementById('variables').innerHTML=''")

    for gdx, gene in enumerate(hof[int(idx)]):
        webview.windows[0].evaluate_js(
            "document.getElementById('variables').innerHTML += '<div class=\"pr-5 pb-2 text-sm text-gray-400\"><i>" +
            list(appData["game_variables"].keys())[gdx] + ":</i> " + str(gene) + "</div>'")

    # Update Statistics Distributions
    webview.windows[0].evaluate_js("document.getElementById('distributions').innerHTML=''")

    for key in selected_distributions.keys():
        webview.windows[0].evaluate_js(
            "document.getElementById('distributions').innerHTML += '<div class=\"pr-5 pb-2 text-sm text-gray-400\"><i>" + key + ":</i></div>'")

        fig, ax = plt.subplots()


        ax.hist(selected_distributions[key], color='blue', edgecolor='black', bins=20)
        ax.set(xlabel=key, ylabel="Frequency", title=key+' Distribution: ')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        #plt.legend()

        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

        webview.windows[0].evaluate_js(
            "document.getElementById('distributions').innerHTML += '<img class=\"w-1/2\" src=\"data:image/jpeg;base64, " +str(
            my_base64_jpgData) + "\">'")

        plt.close(fig)

def evolve_simulation():
    global runningEvolution
    global runonce
    global hof
    global nodeInfo
    global startProcessing
    global remainingTasks
    global hostTime
    global experiment_name

    if runningEvolution:
        return
    else:
        runningEvolution = True

    webview.windows[0].load_url(webview.windows[0].get_current_url().replace("show_project", "run_evolution"))

    try:
        # res_json = json.loads(call_simulator(display=False))
        # print(res_json)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if not appData['evolution_parameters']['Seed']:
            appData['evolution_parameters']['Seed'] = timestamp

        if (runonce == False):
            creator.create("FitnessMaximum", base.Fitness, weights=(1.0,1.0))
            creator.create("Individual", list, fitness=creator.FitnessMaximum)
            runonce = True

        toolbox = base.Toolbox()

        print("Registering Chromosome")

        # Register game variables as chromosome
        genes = []
        limits = []
        count = 0
        for key, value in appData['game_variables'].items():
            print("Registering Gene " + key + '_gene')
            if "integer" in value['type']:
                toolbox.register(str(count) + '_gene', random.randint, value['min'], value['max'])
            else:
                toolbox.register(str(count) + '_gene', random.uniform, value['min'], value['max'])
            genes.append(getattr(toolbox, str(count) + '_gene'))
            limits.append(value)
            count += 1

        toolbox.register("individual", tools.initCycle, creator.Individual, genes, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", mutateUniform, indpb=appData['evolution_parameters']['Mutation-Rate'], limits=limits)
        toolbox.register("select", tools.selNSGA2, k=2, nd='standard')
        #toolbox.register("select", tools.selTournament, tournsize=2)

        # Initial population
        population = toolbox.population(n=appData['evolution_parameters']['Population'])

        # Calculate fitness of initial population
        results = []
        configs = []
        for idx, off in enumerate(population):
            progres = str(((0 * len(population)) + idx) / (
                    appData['evolution_parameters']['Generations'] * len(population)) * 100)
            webview.windows[0].evaluate_js(
                "document.getElementById('evolution-progress-bar').style.width='" + progres + "%';")

            config = copy.deepcopy(appData)
            counter = 0
            for key, value in config['game_variables'].items():
                config['game_variables'][key] = off[counter]
                counter += 1

            config['id'] = idx

            configs.append(config)

        if len(list(nodeInfo.keys())) == 0:
            pool = ThreadPool(processes=len(configs))
            async_results = [pool.apply_async(call_simulator, (False, config)) for config in configs]

            totalTime = 0
            currentResults = []
            for idx, async_result in enumerate(async_results):
                response, elapsedTime = async_result.get()
                try:
                    result = json.loads(response)
                except Exception as e:
                    print(response)
                    print(e)
                    raise e
                # result = json.loads(response)
                results.append(result)

                totalTime += elapsedTime

                off = population[configs[idx]["id"]]

                objective_keys = list(result['objectives'].keys())
                fitness2 = (result['objectives'][objective_keys[0]], result['objectives'][objective_keys[1]])
                # fitness = (result['objectives'][objective_keys[0]],)
                off.fitness.values = fitness2

                if (idx == 0):
                    print(fitness2)

                currentResults.append(np.array([fitness2[0], fitness2[1]]))

                setattr(off, 'gen', 0)
                setattr(off, 'stats', result['scenario_statistics'])
                setattr(off, 'distributions', result['scenario_statistics_distribution'])

            hostTime = totalTime / len(async_results)
        else:
            hostCores = multiprocessing.cpu_count()
            hostTasks = []

            for config in configs:

                leastFilledNode = nodeInfo[list(nodeInfo.keys())[0]]
                leastFilledRank = 100000

                # find the node with least amount of tasks to cpu cores available
                for key, node in nodeInfo.items():
                    if node['avgTime'] > 0:
                        nodeRank = len(node['tasks']) / node['cores'] * node['avgTime']
                    elif hostTime > 0:
                        nodeRank = len(node['tasks']) / node['cores'] * hostTime
                    else:
                        nodeRank = len(node['tasks']) / node['cores']

                    if nodeRank < leastFilledRank:
                        leastFilledRank = nodeRank
                        leastFilledNode = node

                if hostTime > 0:
                    hostRank = len(hostTasks) / hostCores * hostTime
                else:
                    hostRank = len(hostTasks) / hostCores

                if nodeRank < hostRank:
                    leastFilledNode['tasks'].append(config)
                else:
                    hostTasks.append(config)

            for key, node in nodeInfo.items():
                print(key + " Node Tasks: " + str(len(node['tasks'])))
            print("Host Tasks: " + str(len(hostTasks)))

            pool = ThreadPool(processes=len(hostTasks))
            async_results = [pool.apply_async(call_simulator, (False, config)) for config in hostTasks]

            startProcessing = True

            totalTime = 0
            currentResults = []
            for idx, async_result in enumerate(async_results):
                response, elapsedTime = async_result.get()

                try:
                    result = json.loads(response)
                except Exception as e:
                    print(response)
                    print(e)
                    raise e
                results.append(result)

                totalTime += elapsedTime

                off = population[hostTasks[idx]['id']]

                objective_keys = list(result['objectives'].keys())
                fitness2 = (result['objectives'][objective_keys[0]], result['objectives'][objective_keys[1]])
                # fitness = (result['objectives'][objective_keys[0]],)
                off.fitness.values = fitness2

                currentResults.append(np.array([fitness2[0], fitness2[1]]))

                setattr(off, 'gen', 0)
                setattr(off, 'stats', result['scenario_statistics'])
                setattr(off, 'distributions', result['scenario_statistics_distribution'])

            hostTime = totalTime / len(async_results)

            while True:
                ready = True

                try:
                    for key, node in nodeInfo.items():
                        if len(node['tasks']) > len(node['results']):
                            ready = False
                except Exception as e:
                    print(e)
                    ready = False

                if ready:
                    break

            print("Nodes ready!")

            for key, node in nodeInfo.items():
                for idx, result in enumerate(node['results']):
                    results.append(result)

                    off = population[node['tasks'][idx]['id']]

                    objective_keys = list(result['objectives'].keys())
                    fitness2 = (result['objectives'][objective_keys[0]], result['objectives'][objective_keys[1]])
                    # fitness = (result['objectives'][objective_keys[0]],)
                    off.fitness.values = fitness2

                    currentResults.append(np.array([fitness2[0], fitness2[1]]))

                    setattr(off, 'gen', 0)
                    setattr(off, 'stats', result['scenario_statistics'])
                    setattr(off, 'distributions', result['scenario_statistics_distribution'])

                node['tasks'].clear()
                node['results'].clear()

            startProcessing = False

            if len(remainingTasks) > 0:
                async_results = [pool.apply_async(call_simulator, (False, config)) for config in remainingTasks]

                for idx, async_result in enumerate(async_results):
                    response, elapsedTime = async_result.get()
                    result = json.loads(response)
                    results.append(result)

                    off = population[remainingTasks[idx]['id']]

                    objective_keys = list(result['objectives'].keys())
                    fitness2 = (result['objectives'][objective_keys[0]], result['objectives'][objective_keys[1]])
                    # fitness = (result['objectives'][objective_keys[0]],)
                    off.fitness.values = fitness2

                    currentResults.append(np.array([fitness2[0], fitness2[1]]))

                    setattr(off, 'gen', 0)
                    setattr(off, 'stats', result['scenario_statistics'])
                    setattr(off, 'distributions', result['scenario_statistics_distribution'])

                remainingTasks.clear()

        # Setup hall of fame
        hof = tools.ParetoFront()

        # Setup the folders for the experiment
        if not os.path.exists(appData['folder_path'] + '/experiments'):
            os.makedirs(appData['folder_path'] + '/experiments')

        new_experiment_path = ""

        if experiment_name == "":
            new_experiment_path = appData['folder_path'] + '/experiments/' + appData['name'] + "_" + timestamp
        else:
            new_experiment_path = appData['folder_path'] + '/experiments/' + appData['name'] + "_" + experiment_name +  "_" + timestamp

        if not os.path.exists(new_experiment_path):
            os.makedirs(new_experiment_path)

        pastResults = []

        cases = []

        for case in appData["cases"]:
            case_ind = []
            for key in case.keys():
                case_ind.append(case[key])

            cases.append(case_ind)

        pareto_progress = []

        # Run the evolution process for the configured number of generations
        for gen in range(appData['evolution_parameters']['Generations']):
            if not runningEvolution:
                return

            hof.clear()

            print("Running gen: " + str(gen))
            genStartTime = time.time()

            # Create new offspring
            offspring = algorithms.varAnd(population, toolbox, cxpb=appData['evolution_parameters']['Crossover-Rate'], mutpb=1.0)

            # inject cases
            if (len(cases) > 0 and appData["evolution_parameters"]["Injection Count"] > 0 and ((appData["evolution_parameters"]["Injection Interval"] > 0 and gen % appData["evolution_parameters"]["Injection Interval"] == 0) or (appData["evolution_parameters"]["Injection Interval"] == 0 and gen == 0))):

                offspring[:] = tools.selNSGA2(offspring, len(offspring), 'standard')[::-1]

                similarities = []
                caseRange = []
                for i in range(len(cases)):
                    similarity = 0

                    goatchrom = offspring[len(offspring)-1]

                    for g in range(len(cases[i])):
                        similarity += pow((cases[i][g] - goatchrom[g]), 2)

                    similarity /= (len(goatchrom) * len(goatchrom))
                    similarity = 1 - similarity

                    similarities.append(similarity)
                    caseRange.append(i)

                similarities = np.array(similarities)
                similarities /= similarities.sum()

                if len(similarities) > 0 and not np.isnan(similarities).any():
                    choices = np.random.choice(caseRange, size=appData["evolution_parameters"]["Injection Count"], p=similarities)

                    for i in range(appData["evolution_parameters"]["Injection Count"]):
                        offspring[i][:] = cases[choices[i]][:]


            # Ensure the game variables are all within their limits
            for ind in offspring:
                for i in range(len(ind)):
                    if "integer" in limits[i]['type']:
                        ind[i] = max(limits[i]['min'], min(limits[i]['max'], int(ind[i])))
                    else:
                        ind[i] = max(limits[i]['min'], min(limits[i]['max'], float(ind[i])))


            results = []
            configs = []
            for idx, off in enumerate(offspring):
                progres = str(((gen * len(offspring)) + idx) / (
                        appData['evolution_parameters']['Generations'] * len(offspring)) * 100)
                webview.windows[0].evaluate_js(
                    "document.getElementById('evolution-progress-bar').style.width='" + progres + "%';")

                config = copy.deepcopy(appData)
                counter = 0
                for key, value in config['game_variables'].items():
                    config['game_variables'][key] = off[counter]
                    counter += 1

                config['id'] = idx

                configs.append(config)

            if len(list(nodeInfo.keys())) == 0:
                pool = ThreadPool(processes=len(configs))
                async_results = [ pool.apply_async(call_simulator, (False, config)) for config in configs ]

                totalTime = 0
                currentResults = []
                for idx, async_result in enumerate(async_results):
                    response, elapsedTime = async_result.get()
                    try:
                        result = json.loads(response)
                    except Exception as e:
                        print(response)
                        print(e)
                        raise e
                    #result = json.loads(response)
                    results.append(result)

                    totalTime += elapsedTime

                    off = offspring[configs[idx]["id"]]

                    objective_keys = list(result['objectives'].keys())
                    fitness2 = (result['objectives'][objective_keys[0]],result['objectives'][objective_keys[1]])
                    #fitness = (result['objectives'][objective_keys[0]],)
                    off.fitness.values = fitness2

                    if(idx==0):
                        print(fitness2)

                    currentResults.append(np.array([fitness2[0], fitness2[1]]))

                    setattr(off, 'gen', gen)
                    setattr(off, 'stats', result['scenario_statistics'])
                    setattr(off, 'distributions', result['scenario_statistics_distribution'])

                hostTime = totalTime / len(async_results)
            else:
                hostCores = multiprocessing.cpu_count()
                hostTasks = []

                for config in configs:

                    leastFilledNode = nodeInfo[list(nodeInfo.keys())[0]]
                    leastFilledRank = 100000

                    # find the node with least amount of tasks to cpu cores available
                    for key, node in nodeInfo.items():
                        if node['avgTime'] > 0:
                            nodeRank = len(node['tasks'])/node['cores'] * node['avgTime']
                        elif hostTime > 0:
                            nodeRank = len(node['tasks']) / node['cores'] * hostTime
                        else:
                            nodeRank = len(node['tasks']) / node['cores']

                        if nodeRank < leastFilledRank:
                            leastFilledRank = nodeRank
                            leastFilledNode = node

                    if hostTime > 0:
                        hostRank = len(hostTasks) / hostCores * hostTime
                    else:
                        hostRank = len(hostTasks) / hostCores

                    if nodeRank < hostRank:
                        leastFilledNode['tasks'].append(config)
                    else:
                        hostTasks.append(config)

                for key, node in nodeInfo.items():
                    print(key+" Node Tasks: "+str(len(node['tasks'])))
                print("Host Tasks: "+str(len(hostTasks)))

                pool = ThreadPool(processes=len(hostTasks))
                async_results = [pool.apply_async(call_simulator, (False, config)) for config in hostTasks]

                startProcessing = True

                totalTime = 0
                currentResults = []
                for idx, async_result in enumerate(async_results):
                    response, elapsedTime = async_result.get()

                    try:
                        result = json.loads(response)
                    except Exception as e:
                        print(response)
                        print(e)
                        raise e
                    results.append(result)

                    totalTime += elapsedTime

                    off = offspring[hostTasks[idx]['id']]

                    objective_keys = list(result['objectives'].keys())
                    fitness2 = (result['objectives'][objective_keys[0]], result['objectives'][objective_keys[1]])
                    # fitness = (result['objectives'][objective_keys[0]],)
                    off.fitness.values = fitness2

                    currentResults.append(np.array([fitness2[0], fitness2[1]]))

                    setattr(off, 'gen', gen)
                    setattr(off, 'stats', result['scenario_statistics'])
                    setattr(off, 'distributions', result['scenario_statistics_distribution'])

                hostTime = totalTime / len(async_results)


                while True:
                    ready = True

                    try:
                        for key, node in nodeInfo.items():
                            if len(node['tasks']) > len(node['results']):
                                ready = False
                    except Exception as e:
                        print(e)
                        ready = False

                    if ready:
                        break

                print("Nodes ready!")

                for key, node in nodeInfo.items():
                    for idx, result in enumerate(node['results']):
                        results.append(result)

                        off = offspring[node['tasks'][idx]['id']]

                        objective_keys = list(result['objectives'].keys())
                        fitness2 = (result['objectives'][objective_keys[0]], result['objectives'][objective_keys[1]])
                        # fitness = (result['objectives'][objective_keys[0]],)
                        off.fitness.values = fitness2

                        currentResults.append(np.array([fitness2[0], fitness2[1]]))

                        setattr(off, 'gen', gen)
                        setattr(off, 'stats', result['scenario_statistics'])
                        setattr(off, 'distributions', result['scenario_statistics_distribution'])

                    node['tasks'].clear()
                    node['results'].clear()

                startProcessing = False

                if len(remainingTasks) > 0:
                    async_results = [pool.apply_async(call_simulator, (False, config)) for config in remainingTasks]

                    for idx, async_result in enumerate(async_results):
                        response, elapsedTime = async_result.get()
                        result = json.loads(response)
                        results.append(result)

                        off = offspring[remainingTasks[idx]['id']]

                        objective_keys = list(result['objectives'].keys())
                        fitness2 = (result['objectives'][objective_keys[0]], result['objectives'][objective_keys[1]])
                        # fitness = (result['objectives'][objective_keys[0]],)
                        off.fitness.values = fitness2

                        currentResults.append(np.array([fitness2[0], fitness2[1]]))

                        setattr(off, 'gen', gen)
                        setattr(off, 'stats', result['scenario_statistics'])
                        setattr(off, 'distributions', result['scenario_statistics_distribution'])

                    remainingTasks.clear()

            # New Population
            population = toolbox.select(population + offspring, k=appData['evolution_parameters']['Population'])

            # Update Hall of Fame
            hof.update(population)

            # Update Pareto Front UI
            webview.windows[0].evaluate_js("document.getElementById('pareto-list').innerHTML=''")

            front_fitnesses = []
            for idx, h in enumerate(hof):
                if (len(h.fitness.values)):
                    front_fitnesses.append([h.fitness.values[0], h.fitness.values[1]])

                    option_text = objective_keys[0] + ":</i> " + str(h.fitness.values[0]) + ",     " + objective_keys[
                        1] + ": " + str(h.fitness.values[1])
                    js_script = "document.getElementById('pareto-list').innerHTML += '<option value=\"" + str(
                        idx) + "\" >" + option_text + "</option>'"
                    webview.windows[0].evaluate_js(js_script)

            webview.windows[0].evaluate_js("document.getElementById('pareto-list').value='0'")

            hof_selected(0)

            front_fitnesses = np.array(front_fitnesses)

            currentResults.clear()

            for idx, h in enumerate(population):
                currentResults.append(np.array([h.fitness.values[0], h.fitness.values[1]]))

            # Plot Fitness of past and current population
            newPast = pastResults.copy()
            newPast.extend(currentResults)
            currentResults = np.array(currentResults)

            fig, ax = plt.subplots()

            if len(pastResults) == 0:
                ax.scatter(currentResults[:, 0], currentResults[:, 1], c="red", s=20, label='Current Population')
            else:
                prevResults = np.array(pastResults)
                ax.scatter(prevResults[:, 0], prevResults[:, 1], c="gray", s=10, label='Previous Populations',
                           alpha=0.2)
                ax.scatter(currentResults[:, 0], currentResults[:, 1], c="red", s=20, label='Current Population')

            ax.set(xlabel=appData["objectives"][0], ylabel=appData["objectives"][1],
                   title='Population Fitness Gen: ' + str(gen))

            # xlimit = ax.get_xlim()
            # ylimit = ax.get_ylim()
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)

            plt.tight_layout()
            plt.legend()

            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='jpg')
            my_stringIObytes.seek(0)
            my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

            command = "document.getElementById('evolution-fitness').setAttribute('src', 'data:image/jpeg;base64," + str(
                my_base64_jpgData) + "');"
            webview.windows[0].evaluate_js(command)

            if (gen == appData['evolution_parameters']['Generations'] - 1):
                fig.savefig(new_experiment_path + "/fitness.png")

            plt.close(fig)

            pastResults = newPast

            # Plot Pareto Front
            fig, ax = plt.subplots()

            for i in range(len(pareto_progress)):
                if i == 0:
                    ax.plot(pareto_progress[i][:, 0], pareto_progress[i][:, 1], c="red", linestyle='--', marker='o',
                            label='Pareto Front: Gen ' + str(0))
                elif i == 1:
                    ax.plot(pareto_progress[i][:, 0], pareto_progress[i][:, 1], c="green", linestyle='--', marker='o',
                            label='Pareto Front: Gen ' + str(
                                int(appData['evolution_parameters']['Generations'] * 0.25)))
                elif i == 2:
                    ax.plot(pareto_progress[i][:, 0], pareto_progress[i][:, 1], c="yellow", linestyle='--', marker='o',
                            label='Pareto Front: Gen ' + str(int(appData['evolution_parameters']['Generations'] * 0.5)))
                elif i == 3:
                    ax.plot(pareto_progress[i][:, 0], pareto_progress[i][:, 1], c="purple", linestyle='--', marker='o',
                            label='Pareto Front: Gen ' + str(
                                int(appData['evolution_parameters']['Generations'] * 0.75)))

            ax.plot(front_fitnesses[:, 0], front_fitnesses[:, 1], c="blue", linestyle='--', marker='o',
                    label='Pareto Front: Gen ' + str(gen))

            ax.set(xlabel=appData["objectives"][0], ylabel=appData["objectives"][1],
                   title='Pareto Front')

            # ax.set_xlim(xlimit)
            # ax.set_ylim(ylimit)

            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            plt.tight_layout()
            plt.legend()

            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='jpg')
            my_stringIObytes.seek(0)
            my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

            command = "document.getElementById('evolution-pareto').setAttribute('src', 'data:image/jpeg;base64," + str(
                my_base64_jpgData) + "');"
            webview.windows[0].evaluate_js(command)

            if (gen == appData['evolution_parameters']['Generations'] - 1):
                fig.savefig(new_experiment_path + "/pareto.png")

            plt.close(fig)

            # Keep track of pareto front progress
            if len(pareto_progress) == 0:
                pareto_progress.append(front_fitnesses)
            elif len(pareto_progress) == 1 and gen > int(appData['evolution_parameters']['Generations'] * 0.25):
                pareto_progress.append(front_fitnesses)
            elif len(pareto_progress) == 2 and gen > int(appData['evolution_parameters']['Generations'] * 0.5):
                pareto_progress.append(front_fitnesses)
            elif len(pareto_progress) == 3 and gen > int(appData['evolution_parameters']['Generations'] * 0.75):
                pareto_progress.append(front_fitnesses)

            genElapsedTime = time.time() - genStartTime
            genTimeLeft = (appData['evolution_parameters']['Generations'] - (gen+1)) * genElapsedTime
            webview.windows[0].evaluate_js("document.getElementById('time_remaining').innerHTML = 'Time Remaining: "+str(time.strftime('%H hr %M min %S sec', time.gmtime(genTimeLeft)))+"'")

        webview.windows[0].evaluate_js("document.getElementById('time_remaining').innerHTML = 'Done!'")

        pareto_solutions = {}
        pareto_solutions['results'] = []

        for h in hof:
            if hasattr(h, 'stats'):
                solution = {}
                solution['game_variables'] = {}

                for idx,g in enumerate(h):
                    key = list(appData['game_variables'].keys())[idx]
                    solution['game_variables'][key] = g

                solution['scenario_statistics'] = h.stats
                solution['scenario_statistics_distribution'] = h.distributions

                solution['objectives'] = {}
                for idx, v in enumerate(h.fitness.values):
                    key = appData['objectives'][idx]
                    solution['objectives'][key] = v

                pareto_solutions['results'].append(solution)

        json_pareto = json.dumps(pareto_solutions)

        print(json_pareto)
        path = new_experiment_path + "/pareto.json"
        with open(path, "w") as outfile:
            outfile.write(json_pareto)

        json_evo_params = json.dumps(appData)

        print(json_evo_params)
        path = new_experiment_path + "/params.json"
        with open(path, "w") as outfile:
            outfile.write(json_evo_params)

        runningEvolution = False
        #webview.windows[0].load_url(webview.windows[0].get_current_url())

        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--project', dest='project', help='project to open and run evolution', type=str,
                            default=None)
        parser.add_argument('-c', '--config', dest='config', help='json configuration for experiment', type=str,
                            default=None)
        parser.add_argument('-n', '--name', dest='name', help='name for experiment', type=str,
                            default=None)

        args = parser.parse_args()
        if args.project != None:
            on_closing()

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        runningEvolution = False
        webview.windows[0].load_url(webview.windows[0].get_current_url())

        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--project', dest='project', help='project to open and run evolution', type=str,
                            default=None)
        parser.add_argument('-c', '--config', dest='config', help='json configuration for experiment', type=str,
                            default=None)
        parser.add_argument('-n', '--name', dest='name', help='name for experiment', type=str,
                            default=None)

        args = parser.parse_args()
        if args.project != None:
            on_closing()

def remove_data(param, key, scrollY):
    global appData
    global scrollPosY
    scrollPosY= scrollY

    try:
        if isinstance(appData[param], list):
            print(appData[param])
            print(key)
            appData[param].remove(key)
        elif isinstance(appData[param], dict):
            del appData[param][key]
    except Exception as e:
        print(e)

    webview.windows[0].load_url(webview.windows[0].get_current_url())

def update_key(param, old_key, new_key, scrollY):
    global appData
    global scrollPosY
    scrollPosY = scrollY

    appData[param][new_key] = appData[param].pop(old_key)

    webview.windows[0].load_url(webview.windows[0].get_current_url())

def update_array(param, index, value, scrollY):
    global appData
    global scrollPosY
    scrollPosY = scrollY

    print(index)
    print(int(index))
    appData[param][int(index)] = value
    print(appData)

    webview.windows[0].load_url(webview.windows[0].get_current_url())

def update_project_info(key, value, scrollY):
    print(str(key)+ " : "+str(value))

    global appData
    global scrollPosY
    scrollPosY = scrollY

    appData[key] = value
    print(appData)

    webview.windows[0].load_url(webview.windows[0].get_current_url())

def add_data(key, scrollY):
    global appData
    global paramCount
    global scrollPosY
    scrollPosY = scrollY

    try:
        if isinstance(appData[key], list):
            appData[key].append("default")
        elif isinstance(appData[key], dict):
            paramCount += 1
            if "game_variables" in key:
                appData[key]["new param"+str(paramCount)] = { 'value': 0, 'min': 0, 'max': 0, 'type': 'integer' }
            else:
                appData[key]["new param" + str(paramCount)] = 0
    except Exception as e:
        print(e)

    #webview.windows[0].evaluate_js("var y = window.scrollY; location.reload(); window.scrollTo(0, y);")
    webview.windows[0].load_url(webview.windows[0].get_current_url())

def empty_project():
    mutex.acquire()
    try:
        global appOpen
        if appOpen:
            return
        else:
            appOpen = True
    finally:
        mutex.release()

    global  appData
    global paramCount
    appData = {}
    appData['name'] = "untitled"
    appData['folder_path'] = os.path.abspath(join("projects", "untitled"))
    appData['simulator_path'] = os.path.abspath(join("projects", "untitled"))
    appData['simulator_command'] = ""
    appData['evolution_parameters'] = {
        'Seed': '',
        'Generations': 100,
        'Population': 64,
        'Mutation-Rate': 0.1,
        'Crossover-Rate': 0.9,
        'Simulation Count': 1000,
        "Injection Count": 1,
        "Injection Interval": 5
    }
    appData['game_variables'] = {}
    appData['cases'] = [ ]
    appData['scenarios'] = [ ]
    appData['scenario_parameters'] = {}
    appData['scenario_statistics'] = []
    appData['objectives'] = [
        'objective1',
        'objective2'
    ]
    paramCount = 0

    # Get smallest screen dimensions
    minWidth = 10000
    minHeight = 10000

    for screen in webview.screens:
        if screen.width < minWidth:
            minWidth = screen.width

        if screen.height < minHeight:
            minHeight = screen.height

    minWidth -= 50
    minHeight -= 50

    if minWidth > 1300:
        minWidth = 1300

    if minHeight > 1100:
        minHeight = 1100

    window = webview.create_window("New Project - Game Disruptor", app, frameless=False, easy_drag=True,
                                   on_top=False,
                                   confirm_close=True, width=minWidth, height=minHeight, background_color='#1F2326')
    window.load_url(window.get_current_url() + 'new_project')

    window.closing += on_closing
    window.expose(close, case_selected, add_case, remove_case, add_data, update_key, update_project_info, remove_data, on_update, on_update_scenario, update_array, create_project, show_simulation, evolve_simulation, stop_evolving, hof_selected, pareto_selected, view_experiment, edit_existing_project, cancel_edit, save_project, host, connectToHost, select_scenario)
    webview.windows[0].destroy()

    appOpen = False

def on_closing():
    mutex.acquire()
    try:
        os._exit(1)
    finally:
        quit()
    mutex.release()

def create_project():
    path = appData['folder_path'] + "/config.json"

    if not os.path.exists(appData['folder_path']):
        os.makedirs(appData['folder_path'])

    if not os.path.exists(join(appData['folder_path'], 'experiments')):
        os.makedirs(join(appData['folder_path'], 'experiments'))

    with open(path, "w") as outfile:
        outfile.write(json.dumps(appData))

    webview.windows[0].load_url(webview.windows[0].get_current_url().replace("new_project", "show_project"))

def on_update_scenario(param, key, s):
    mutex.acquire()
    try:
        try:
            if isinstance(appData[param][key], int):
                appData[param][key] = int(s)
            elif isinstance(appData[param][key], float):
                appData[param][key] = float(s)
            else:
                appData[param][key] = s
        except:
            if not s:
                webview.windows[0].evaluate_js("alert('"+str(key)+" cannot be empty!');")
                print(param + " " + key + " cannot be left blank!")
            else:
                webview.windows[0].evaluate_js("alert('"+s+" is not valid input for "+str(key)+"');")
                print(s + " cannot be converted into " + str(type(appData[param][key])) + " for " + param + " " + key + "!")
            webview.windows[0].evaluate_js("document.getElementById('input_" + key + "').value = '" + str(appData[param][key]) + "';")

        if "show_project" in webview.windows[0].get_current_url():
            path = appData['folder_path'] + "/config.json"
            with open(path, "w") as outfile:
                outfile.write(json.dumps(appData))

        print(appData)
    finally:
        mutex.release()

def on_update(param, key, p, s):
    mutex.acquire()
    try:
        try:
            if isinstance(appData[param][key][p], int):
                appData[param][key][p] = int(s)
            elif isinstance(appData[param][key][p], float):
                appData[param][key][p] = float(s)
            else:
                appData[param][key][p] = s
        except:
            if not s:
                webview.windows[0].evaluate_js("alert('"+str(key)+" cannot be empty!');")
                print(param + " " + key + " cannot be left blank!")
            else:
                webview.windows[0].evaluate_js("alert('"+s+" is not valid input for "+str(key)+"');")
                print(s + " cannot be converted into " + str(type(appData[param][key][p])) + " for " + param + " " + key + "!")
            webview.windows[0].evaluate_js("document.getElementById('input_" + key + "_" + p +"').value = '" + str(appData[param][key][p]) + "';")

        if "min" in p and appData[param][key]['min'] > appData[param][key]['max']:
            appData[param][key][p] = 0
            webview.windows[0].evaluate_js("alert('" + str(key) + " min cannot be greater than max!');")
            print(param + " " + key + " min cannot be greater than max!")
            webview.windows[0].evaluate_js("document.getElementById('input_" + key + "_min').value = '0';")

        if "min" in p or "max" in p:
            appData[param][key]['value'] = 0

        print(appData)

        if "show_project" in webview.windows[0].get_current_url():
            path = appData['folder_path'] + "/config.json"
            with open(path, "w") as outfile:
                outfile.write(json.dumps(appData))
    finally:
        mutex.release()

def select_scenario(scenario):
    global appData

    appData['scenarios'].remove(scenario)
    appData['scenarios'].insert(0, scenario)

    if "show_project" in webview.windows[0].get_current_url():
        path = appData['folder_path'] + "/config.json"
        with open(path, "w") as outfile:
            outfile.write(json.dumps(appData))

def add_case():
    global appData

    case_vars = {}
    for key, game_var in appData['game_variables'].items():
        case_vars[key] = game_var["value"]

    appData["cases"].append(case_vars)

    webview.windows[0].load_url(webview.windows[0].get_current_url())

def open_project(project):
    print(project)

    mutex.acquire()
    try:
        global appOpen
        if appOpen:
            return
        else:
            appOpen = True
    finally:
        mutex.release()

    global experiment_name

    # Get smallest screen dimensions
    minWidth = 10000
    minHeight = 10000

    for screen in webview.screens:
        if screen.width < minWidth:
            minWidth = screen.width

        if screen.height < minHeight:
            minHeight = screen.height

    minWidth -= 50
    minHeight -= 50

    if minWidth > 1300:
        minWidth = 1300

    if minHeight > 1100:
        minHeight = 1100

    global appData

    project = json.loads(project)
    project_path = project["folder_path"]
    print(join(project_path, 'config.json'))
    with open(join(project_path, 'config.json')) as json_file:
        appData = json.load(json_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', dest='project', help='project to open and run evolution', type=str,
                        default=None)
    parser.add_argument('-c', '--config', dest='config', help='json configuration for experiment', type=str,
                        default=None)
    parser.add_argument('-n', '--name', dest='name', help='name for experiment', type=str,
                        default=None)

    args = parser.parse_args()
    if args.project != None and args.config != None:
        print(args.config)
        tempConfig = json.loads(args.config)

        for key in tempConfig.keys():
            for key2 in tempConfig[key].keys():
                appData[key][key2] = tempConfig[key][key2]

    window = webview.create_window(appData['name']+" - Game Disruptor", app, frameless=False, easy_drag=True, on_top=False,
                                   confirm_close=True, width=minWidth, height=minHeight, background_color='#1F2326')
    window.load_url(window.get_current_url() + 'show_project')

    window.closing += on_closing
    window.expose(close, case_selected, add_case, remove_case, add_data, update_key, remove_data, on_update, update_project_info, on_update_scenario, update_array, show_simulation, evolve_simulation, stop_evolving, hof_selected, pareto_selected, view_experiment, edit_existing_project, cancel_edit, save_project, host, connectToHost, select_scenario)
    webview.windows[0].destroy()

    appOpen = False

    # Start evolution automatically
    if args.project != None:
        if args.name != None:
            experiment_name = args.name
        else:
            experiment_name = ""

        evolve_simulation()

def close():
    print("Quit!")
    on_closing()

def go_landing(window):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', dest='project', help='project to open and run evolution', type=str, default=None)
    parser.add_argument('-c', '--config', dest='config', help='json configuration for experiment', type=str, default=None)
    parser.add_argument('-n', '--name', dest='name', help='name for experiment', type=str, default=None)

    args = parser.parse_args()
    if(args.project != None):
        print("Automatically open: "+str(args.project))

        config_files = [join("projects", f, "config.json") for f in listdir("projects") if
                        not isfile(join("projects", f))]

        for path in config_files:
            with open(path) as json_file:
                j = json.load(json_file)
                j['folder_path'] = os.path.abspath(os.path.dirname(path))
                print(j['folder_path'])

                if args.project.lower() in j['name'].lower():
                    open_project(json.dumps(j))
                    return
    else:
        window.load_url(window.get_current_url() + "landing")



def on_ready():
    webview.windows[0].loaded -= on_ready
    try:
        import pyi_splash

        pyi_splash.update_text('UI Loaded ...')
        pyi_splash.close()
    except:
        pass

if __name__ == '__main__':
    # t = threading.Thread(target=start_server)
    # t.daemon = True
    # t.start()

    # Get smallest screen dimensions
    minWidth = 10000
    minHeight = 10000

    for screen in webview.screens:
        if screen.width < minWidth:
            minWidth = screen.width

        if screen.height < minHeight:
            minHeight = screen.height

    minWidth -= 50
    minHeight -= 50

    if minWidth > 800:
        minWidth = 800

    if minHeight > 500:
        minHeight = 500


    window = webview.create_window("Welcome to the Game Disruptor", app, frameless=True, easy_drag=False, on_top=False, confirm_close=False, width=minWidth, height=minHeight, background_color='#1F2326')
    #window.closing += on_closing
    window.loaded += on_ready
    window.expose(close, open_project, empty_project, on_update)
    webview.start(go_landing, window)
    sys.exit()