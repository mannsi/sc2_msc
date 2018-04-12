import argparse
import time

time.sleep(5)

print("Done sleeping, lets add a file to cloud storage")

parser = argparse.ArgumentParser()
parser.add_argument('--tb_output', dest='tb_output', type=str)
args = parser.parse_args()

print("got arg ", args.tb_output)

with open(args.tb_output + '/from_python.txt', 'w') as f:
    f.write("text from python")

