from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import json, os

app = Flask(__name__, static_folder='dashboard')
CORS(app)
ALERT_LOG = 'alerts.json'

@app.route('/')
def index():
    return send_from_directory('dashboard', 'index.html')

@app.route('/api/alerts')
def get_alerts():
    if not os.path.exists(ALERT_LOG): return jsonify([])
    with open(ALERT_LOG) as f:
        try: data = json.load(f)
        except: data = []
    return jsonify(data[-100:])

@app.route('/api/stats')
def get_stats():
    if not os.path.exists(ALERT_LOG):
        return jsonify({'total':0,'high':0,'medium':0})
    with open(ALERT_LOG) as f:
        try: data = json.load(f)
        except: data = []
    return jsonify({
        'total':  len(data),
        'high':   sum(1 for a in data if a.get('severity')=='HIGH'),
        'medium': sum(1 for a in data if a.get('severity')=='MEDIUM')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)