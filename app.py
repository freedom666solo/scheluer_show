from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from kubernetes.client import ApiException
import json
import client_util
from datetime import datetime, timedelta
import predict
import pandas as pd

app = Flask(__name__)
CORS(app)
@app.route('/api/form', methods=['POST'])
def handle_form():
    data = request.json
    print('接收到的数据:',data)
    # 这里可以添加处理数据的逻辑，例如保存到数据库
    deployname=data['deployname']
    if(deployname==''):
        print('部署名称不能为空')
        return jsonify({'message': '部署名称不能为空'}), 400
    namespace=data['namespace']
    if(deployname==''):
        print('命名空间不能为空')
        return jsonify({'message': '命名空间不能为空'}), 400
    namespaces=client_util.get_ns()
    if (namespace not in namespaces):
        print('命名空间不存在')
        return  jsonify({'message': '命名空间不存在'}), 400
    replicas=(data['replicas'])
    try:
        rpl=int(replicas)
        if(rpl<1 or rpl>10):
            print('数字错误')
            return jsonify({'message': '数字错误'}), 400
    except:
        print('replica格式错误，必须为整数')
        return jsonify({'message': 'replica格式错误，必须为整数'}), 400
    container_name=data['container_name']
    image_name=data['image_name']
    if(container_name==''):
        print('容器名称不能为空')
        return jsonify({'message': '容器名称不能为空'}), 400
    if(image_name==''):
        print('镜像名称不能为空')
        return jsonify({'message': '镜像名称不能为空'}), 400
    image_version=data['image_version']
    if(image_version==''):
        image_version='latest'
    image_total=image_name+':'+image_version
    scheduler=data['scheduler']
    scheduler_positon=data['scheduler_positon']
    use_gpu=str(data['use_gpu'])
    cpu_request=data['cpu_request']
    mem_request = data['mem_request']
    gpu_request = data['gpu_request']
    cpu_limit = data['cpu_limit']
    gpu_limit = data['gpu_limit']
    mem_limit=data['mem_limit']


    resp,code=client_util.deploy(deployname, namespace, replicas, container_name,
           image_total, scheduler, scheduler_positon, use_gpu, cpu_request, mem_request, gpu_request, cpu_limit,
           gpu_limit, mem_limit)
    if(code==200):
        return jsonify({'message': "容器/"+container_name+"创建成功"}), 200
    else:
        return jsonify({'message': resp}), 400

@app.route('/api/pretem', methods=['GET'])
def tem_predict():
    data = predict.predict()
    json_data = {}
    for index, row in data.iterrows():
        json_data[str(row['time'])] = str(row['temperature'])
    return jsonify(json_data), 200

@app.route('/api/helmet', methods=['GET'])
def helmet_recognize():
    image_path = './detect.jpg'
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/api/curtem', methods=['GET'])
def get_tem():
    data = pd.read_csv('newdata.csv')
    df = data['Indoor_temperature_room'].iloc[-31:].reset_index(drop=True)
    print(df)
    current_time = datetime.now().replace(second=0, microsecond=0)
    time_adjusted_data = {}
    for i in range(len(df)):
        time_adjusted = current_time - timedelta(minutes=30 - i)
        time_adjusted_data[time_adjusted.strftime('%Y-%m-%d %H:%M:%S')] = df[i]


    # 返回 JSON 数据
    return jsonify(time_adjusted_data),200

@app.route('/api/index', methods=['GET'])
def get_index():
    df = pd.read_csv('newdata.csv')
    df = df.drop(columns=['dateTime', 'Indoor_temperature_room'])
    # 获取最后一行数据
    last_row = df.iloc[-1].to_dict()

    # 转换为 JSON 格式
    return jsonify(last_row)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
