"""
Webapp
"""

import glob
import flask
import numpy as np

from main import *

app = flask.Flask(__name__)

seed_val = 42
np.random.seed(seed_val)

task = args = dmn = network_name = test_input = test_q = test_answer = pred_answer = pred_prob = memory_probs = None

@app.route('/')
def index():
    return flask.render_template("index.html")

@app.route('/get/taskType', methods=['GET'])
def get_taskType():
    global task, args, dmn, network_name, test_input, test_q, test_answer, pred_answer, pred_prob, memory_probs

    task = flask.request.args.get('taskType_i')
    task = ''.join(task.split())

    args = dmn_start()

    if task == '2':
        args.babi_id = '2'
        args.load_state = 'states/dmn_smooth.mh3.n80.bs10.d0.01.babi2.epoch129.test0.96979.state'
    elif task == '3':
        args.babi_id = '3'
        args.load_state = 'states/dmn_smooth.mh3.n80.bs10.d0.01.babi3.epoch5.test2.28914.state'
    elif task == '6':
        args.babi_id = '6'
        args.load_state = 'states/dmn_smooth.mh3.n80.bs10.d0.01.babi6.epoch90.test0.01423.state'
    elif task == '17':
        args.babi_id = '17'
        args.load_state = 'states/dmn_smooth.mh3.n80.bs10.d0.1.babi17.epoch498.test1.19528.state'
    else: 
        args.babi_id = '2'
        args.load_state = 'states/dmn_smooth.mh3.n80.bs10.d0.01.babi2.epoch129.test0.96979.state'

    args, network_name, dmn = dmn_mid(args)

@app.route('/get/story', methods=['GET'])
def get_story():
    global task, args, dmn, network_name, test_input, test_q, test_answer, pred_answer, pred_prob, memory_probs

    test_input = dmn.test_input 
    test_q = dmn.test_q
    test_answer = dmn.test_answer
    
    question_idx  = np.random.randint(test_q.shape[0])

    return_data = dmn.step(question_idx, 'test')

    story_txt = [dmn.ivocab[k] for i in return_data["inp"] for j in i for k in j]
    question_txt = [dmn.ivocab[j] for i in return_data["q"] for j in i]
    correct_answer = [dmn.ivocab[i] for i in return_data["answers"]]
    pred_answer = [dmn.ivocab[i] for i in return_data["prediction"].argmax(axis=1)]
    pred_prob = [return_data["prediction"][0][i] for i in return_data["prediction"].argmax(axis=1)]

    story_txt = ' '.join(filter(None, story_txt))
    question_txt = ' '.join(filter(None, question_txt))
    correct_answer = correct_answer[0]
    pred_answer = pred_answer[0]
    pred_prob = float(pred_prob[0])

    memory_probs = return_data["attentions"][0]

    story_txt = story_txt.replace(" . ", ".\n")
    question_txt += "?"

    return flask.jsonify({
        "question_idx": question_idx,
        "story": story_txt,
        "question": question_txt,
        "correct_answer": correct_answer, 
        #"pred_answer" : pred_answer
    })
    
@app.route('/get/answer', methods=['GET'])
def get_answer():
    question_idx  = flask.request.args.get('question_idx')
    user_question = flask.request.args.get('user_question', '')

    return flask.jsonify({
        "pred_answer" : pred_answer,
        "pred_prob" : pred_prob,
        "memory_probs": memory_probs.T.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
