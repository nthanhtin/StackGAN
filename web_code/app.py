from flask import Flask, render_template, request
import shutil
import os
import time

app = Flask(__name__,template_folder='.')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/getimage", methods=['GET', 'POST'])
def get_img():
	print("in function1")
	textToSearch = request.get_data(as_text=True)
	print(textToSearch)
	textFile=open('textFile.txt', 'w')
	textFile.write(textToSearch)
	textFile.close()
	sudoPassword = "Tinthanh1"
	cmd = "cp ./textFile.txt ../StackGAN/Data/birds/example_captions.txt"
	os.system('echo %s|sudo -S %s' % (sudoPassword, cmd))
	fileName = './static/*.jpg'
	cmd = "rm -rf " + str(fileName)
	os.system(cmd)
	cmd = "nvidia-docker run -it -v ~/StackGAN:/root/StackGAN --entrypoint sh nthanhtin/stackgan:latest -al /root/birds_demo.sh"
	os.system(cmd)
	image_name = "birds"+str(time.time_ns())+".jpg"
	status = shutil.copyfile("../StackGAN/Data/birds/example_captions/sentence0.jpg", "./static/"+image_name)
	return image_name


if __name__ == '__main__':
    app.run(debug=True)