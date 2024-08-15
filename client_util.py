import json

import kubernetes as k8s
from kubernetes import client, watch
from kubernetes.client import ApiException

k8s.config.load_kube_config()
v1 = client.CoreV1Api()
api_client = client.ApiClient()
k8s_apps_v1=client.AppsV1Api()
import yaml
def get_ns():
    list=[]
    for ns in v1.list_namespace().items:
        list.append(ns.metadata.name)
    list.append("default")
    return list

def deploy(deployname,namespace, replicas,container_name,
           image_total,scheduler,scheduler_positon,use_gpu, cpu_request,mem_request,gpu_request,cpu_limit,gpu_limit,mem_limit):
    with open('exa.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config['metadata']['name']=deployname
    config['metadata']['namespace'] = namespace
    config['metadata']['labels']['app']=deployname
    config['spec']['selector']['matchLabels']['app']=deployname
    config['spec']['template']['metadata']['labels']['app']=deployname
    config['spec']['replicas']=int(replicas)
    config['spec']['template']['spec']['containers'][0]['name']=container_name
    config['spec']['template']['spec']['containers'][0]['image']=image_total
    if(scheduler=="k8s-default"):
        del config['spec']['template']['spec']['schedulerName']
    if(use_gpu=='true' and scheduler_positon!='edge' and gpu_limit!='' and gpu_request!= ''):
        config['spec']['template']['metadata']['labels']['needgpu'] = "true"
        config['spec']['template']['spec']['containers'][0]['resources']['limits']['aliyun.com/gpu-mem']=int(gpu_limit[:-1])
        config['spec']['template']['spec']['containers'][0]['resources']['requests']['aliyun.com/gpu-mem'] = int(gpu_request[:-1])
    elif(use_gpu=='true'):
        config['spec']['template']['metadata']['labels']['needgpu'] = "true"
        del config['spec']['template']['spec']['containers'][0]['resources']['limits']['aliyun.com/gpu-mem']
        del config['spec']['template']['spec']['containers'][0]['resources']['requests']['aliyun.com/gpu-mem']
    else:
        config['spec']['template']['metadata']['labels']['needgpu'] = "false"
        del config['spec']['template']['spec']['containers'][0]['resources']['limits']['aliyun.com/gpu-mem']
        del config['spec']['template']['spec']['containers'][0]['resources']['requests']['aliyun.com/gpu-mem']

    config['spec']['template']['spec']['containers'][0]['resources']['limits']['cpu']=cpu_limit
    config['spec']['template']['spec']['containers'][0]['resources']['limits']['memory'] = mem_limit
    config['spec']['template']['spec']['containers'][0]['resources']['requests']['cpu'] = cpu_request
    config['spec']['template']['spec']['containers'][0]['resources']['requests']['memory'] = mem_request
    if(scheduler_positon=="edge"):
        config['spec']['template']['metadata']['labels']['appointedge'] = "true"
    else:
        config['spec']['template']['metadata']['labels']['appointedge'] = "false"
    try:
        # 尝试创建 Deployment
        resp = k8s_apps_v1.create_namespaced_deployment(body=config, namespace=namespace)
        print(resp)
        return resp,200
    except ApiException as e:
        # 捕获 ApiException 并返回详细错误信息
        if e.body:
            error_json = json.loads(e.body)
            error_message = error_json.get("message", "No error message provided")
        return error_message,400


def test():
    with open('exa.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(config)


    pass
if __name__=="__main__":
    test()


