### Introductions to some modules.

- [main.py](https://github.com/CVBox/PytorchCV/tree/master/main.py)
    - The entry of the whole project.
    - You can specify some arguments through the command line.
    - Launch the Logger & Configer.
    - Init the model & select the phase of the task you selected.

- [configer.py](https://github.com/CVBox/PytorchCV/tree/master/utils/tools/configer.py)
    - It is used for the maintenance of all the parameters that passed by the command line and hypes file.
    - The priority of the command line is higher than the priority of the hypes file.
    - Pass message to every corner of the framework.
    - It's very very very important!!!
    
- [datasets](https://github.com/CVBox/PytorchCV/tree/master/datasets)
    - Task Loader for each Task Type, and each method have a corresponding data loader.
    - Transform operations, aug_transform is used for data augmentation that is available to all task type. task-related transforms are only for the specified task type.
    - the generator file in the directory named for dataset is the script that processes the data into the defined format.
    
- [hypes](https://github.com/CVBox/PytorchCV/tree/master/hypes)
    - Each sub project will have a corresponding hype file that contains all the parameters for the project. 
    - You could change some parameters through the command line or api for the configer, such as "add_value".

- [methods](https://github.com/CVBox/PytorchCV/tree/master/methods)
    - This directory that contains all the train, val, test & deploy scripts for each project.
    - You could specify the phase of project through command line.
    - It is also available for the programmers to use the class for train, test & deploy.

- [loss](https://github.com/CVBox/PytorchCV/tree/master/loss)
    - This directory that contains all loss for each task type.
    - You could use all the loss through the correponding loss manager.

- [models](https://github.com/CVBox/PytorchCV/tree/master/models)
    - This directory that contains all models for each task type.
    - You could load pretrained backbone models by simply specify the pretrained model path through the command line.

- [utils](https://github.com/CVBox/PytorchCV/tree/master/utils)
    - utils.layers contains all the packaged layers that are easy to use. 
    - utils.tools contained some tool classes that benefit to the whole project.

- [val](https://github.com/CVBox/PytorchCV/tree/master/val)
    - This directory that contains all the validation scripts for each task.

- [vis](https://github.com/CVBox/PytorchCV/tree/master/vis)
    - This directory that contains all the visualization scripts for each task.
    - Parsers are used to render the label files.
    - Visualizers are used to visualize the results during the trainig, testing, and deploying.

- [imagesite](https://github.com/CVBox/PytorchCV/tree/master/imagesite)
    - ImageSite is used to look through the image files on remote hosts without mounting or scp.
    - Furthermore, we can also submit command of PytorchCV, check the results of checkpoints, submit the results to evaluation server, and so on.