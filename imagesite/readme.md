### imageSite
ImageSite is used to look through the image files on remote hosts without mounting or scp. Furthermore, we can also submit command of PytorchCV, check the results of checkpoints, submit the results 
to evaluation server, and so on.

### Usage

- Change the directory of project and datasets in config.py and imagesite.conf.
- Install the required packages, such as Nginx, Flask, gunicon. Also copy the conf file to the right directory. Just run the shell script "install.sh".
- Run the Nginx server & Flask backend framework, just run the shell script "run-site.sh".
- You will see the imagesite through your web browser. The site url is "http://ip-adress/imagesite".

