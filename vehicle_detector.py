from VehicleDetectionPipeline import *
from moviepy.editor import VideoFileClip
import sys
if __name__=='__main__':
	arg_count =len(sys.argv)
	#print(arg_count)
	if arg_count!=3:
		print('Run this script as python vehicle_detector.py your_input_file.mp4 your_output_file.mp4')
	else:
		try:
			pipeline = VehicleDetectionPipeline()			
			inputfile = str(sys.argv[1])
			outputfile = str(sys.argv[2])
			
			clip1 = VideoFileClip(inputfile)
			white_clip = clip1.fl_image(pipeline.process) #NOTE: this function expects color images!!
			white_clip.write_videofile(outputfile, audio=False)
		except Exception as e:
			print(str(e))
