import argparse

from azureml.core import Workspace, ComputeTarget, Experiment
from azureml.core.compute import AmlCompute
from azureml.core.model import Model
import os


def init_workspace_data(data_folder):
    ws = Workspace.from_config()
    print('Initializing data for workspace', ws.name, sep = '\t')
    ds = ws.get_default_datastore()
    print('Uploading to', ds.datastore_type, ds.account_name, ds.container_name)
    ds.upload(src_dir=data_folder, target_path='mnist', overwrite=True, show_progress=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
    parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
    return parser.parse_args()

def get_remote_compute():
    ws = Workspace.from_config()
    return prepare_remote_compute(ws)

def prepare_remote_compute(ws):
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")
    compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 1)
    compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

    # This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")


    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print('found compute target. Using it. ' + compute_name)
    else:
        print('creating a new compute target...')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                    min_nodes = compute_min_nodes,
                                                                    max_nodes = compute_max_nodes)
        # create the cluster
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

        # can poll for a minimum number of nodes and for a specific timeout.
        # if no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

        # For a more detailed view of current AmlCompute status, use get_status()
        print(compute_target.get_status().serialize())

    return compute_target


def best_accuracy(model_name):
    best = 0.0
    models = Model.list(workspace=Workspace.from_config(), name=model_name)
    for model in models:
        if 'accuracy' in model.properties:
            accuracy = float(model.properties.get('accuracy'))
            if accuracy > best:
                best = accuracy
    return best
