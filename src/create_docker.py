import os
from azureml.core.runconfig import CondaDependencies
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import ContainerImage, Image


# This is intended to be run from Azure Devops after a trigger from model update


ws = Workspace.from_config()
model_name = 'sklearn_mnist'


def get_best_model(model_name):
    best_model = None
    best = 0.0
    models = Model.list(workspace=Workspace.from_config(), name=model_name)
    for model in models:
        if 'accuracy' in model.properties:
            accuracy = float(model.properties.get('accuracy'))
            if accuracy > best:
                best = accuracy
                best_model = model
    return best_model



def build_container():

    cd = CondaDependencies.create(pip_packages=['azureml-sdk==1.0.39',
                                                'scikit-learn==0.21.1',
                                                'joblib==0.13.2'])

    cd.save_to_file(base_directory='./', conda_file_path='myenv.yml')

    model = get_best_model(model_name)
    print('model',model)

    img_config = ContainerImage.image_configuration(execution_script='score.py',
                                                    runtime='python',
                                                    conda_file='myenv.yml',
                                                    dependencies=['.'])

    image_name = model_name.replace("_", "").lower()

    print("Image name:", image_name)

    image = Image.create(name = image_name,
                         models = [model],
                         image_config = img_config,
                         workspace = ws)

    image.wait_for_creation(show_output = True)


    if image.creation_state != 'Succeeded':
        raise Exception('Image creation status: {image.creation_state}')

    print('{}(v.{} [{}]) stored at {} with build log {}'.format(image.name, image.version, image.creation_state, image.image_location, image.image_build_log_uri))




if __name__ == '__main__':
    build_container()
