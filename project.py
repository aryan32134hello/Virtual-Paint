import mediapipe as mp
import cv2
import numpy as np
import time

# Intializing the Constants

start_x = 150 #Start of the drawingTool image
max_x, max_y = 250+start_x, 50 #End coordinates of draw_tool image
curr_tool = "select tool"  #Default value to show when no tool is selected
time_init = True 
choose_rad = 30 
var_inits = False
line_thick = 4  #Thickness of lines
prev_x, prev_y = 0,0 #Initializing the last coordinates
last_timestamp = 0 # Initializing the last recording timestamp

def getTool(x):
	"""
    Function to get the tool as per the x position index finger tip

    @param x:X position of the index finger tip
    @returns: The tool selected according to x position of index finger

    """
	if x < 50 + start_x:
		return "line"

	elif x<100 + start_x:
		return "rectangle"

	elif x < 150 + start_x:
		return"draw"

	elif x<200 + start_x:
		return "circle"

	else:
		return "erase"

def isMiddleRaised(yi, y9):
	"""
    Function to check if middle finger is raised or not

    @param yi: y position of middle finger tip
	@param y9: y position of middle finger mcp
    @returns: true if middle finger is raised else false

    """
	print(y9-yi)
	if (y9 - yi) > 40:
		return True

	return False



hands = mp.solutions.hands #Intializing the hand detection model from mediapipe
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) #Initializing the model probablities
draw = mp.solutions.drawing_utils 


# reading the image of draw_tool
draw_tool = cv2.imread('drawingTool.png')
draw_tool = draw_tool.astype('uint8')

mask = np.ones((480, 640))*255
mask = mask.astype('uint8')

cap = cv2.VideoCapture(0) # Initializing the webcam 
while True:
	success,image = cap.read()
	image = cv2.flip(image,1) #Fliping the image to show
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	results = hand_landmark.process(img) # processing each frame
    # print(results.multi_handedness)
	if results.multi_hand_landmarks:
		for hand_landmarks in results.multi_hand_landmarks:
			draw.draw_landmarks(image, hand_landmarks, hands.HAND_CONNECTIONS) #Drawing landmarks ad landmarks connection on frame
			x, y = int(hand_landmarks.landmark[8].x*640), int(hand_landmarks.landmark[8].y*480) #Getting the coordinates of index tip

			if x < max_x and y < max_y and x > start_x: # To check if the index finger is in the box 
				if time_init:
					cur_time = time.time()
					time_init = False
				last_timestamp = time.time()

				cv2.circle(image, (x, y), choose_rad, (0,255,255), 2)
				choose_rad=choose_rad-1

				if (last_timestamp - cur_time) > 0.8:
					curr_tool = getTool(x) 
					print("your current tool set to : ", curr_tool)
					time_init = True
					choose_rad = 30

			else:
				time_init = True
				choose_rad = 30

			if curr_tool == "draw":
				xi, yi = int(hand_landmarks.landmark[12].x*640), int(hand_landmarks.landmark[12].y*480)
				y9  = int(hand_landmarks.landmark[9].y*480)

				if isMiddleRaised(yi, y9):
					cv2.line(mask, (prev_x, prev_y), (x, y), 0, line_thick) #To draw randomly
					prev_x, prev_y = x, y

				else:
					prev_x = x
					prev_y = y



			elif curr_tool == "line":
				xi, yi = int(hand_landmarks.landmark[12].x*640), int(hand_landmarks.landmark[12].y*480)
				y9  = int(hand_landmarks.landmark[9].y*480)

				if isMiddleRaised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.line(image, (xii, yii), (x, y), (50,152,255), line_thick) #To draw line

				else:
					if var_inits:
						cv2.line(mask, (xii, yii), (x, y), 0, line_thick)
						var_inits = False

			elif curr_tool == "rectangle":
				xi, yi = int(hand_landmarks.landmark[12].x*640), int(hand_landmarks.landmark[12].y*480)
				y9  = int(hand_landmarks.landmark[9].y*480)

				if isMiddleRaised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.rectangle(image, (xii, yii), (x, y), (0,255,255), line_thick) #To draw rectangle

				else:
					if var_inits:
						cv2.rectangle(mask, (xii, yii), (x, y), 0, line_thick) #To draw rectangle
						var_inits = False

			elif curr_tool == "circle":
				xi, yi = int(hand_landmarks.landmark[12].x*640), int(hand_landmarks.landmark[12].y*480)
				y9  = int(hand_landmarks.landmark[9].y*480)

				if isMiddleRaised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.circle(image, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), line_thick) #To draw circle

				else:
					if var_inits:
						cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), line_thick) #To draw circle
						var_inits = False

			elif curr_tool == "erase":
				xi, yi = int(hand_landmarks.landmark[12].x*640), int(hand_landmarks.landmark[12].y*480)
				y9  = int(hand_landmarks.landmark[9].y*480)

				if isMiddleRaised(yi, y9):
					cv2.circle(image, (x, y), 30, (0,0,0), -1)
					cv2.circle(mask, (x, y), 30, 255, -1)



	frame = cv2.bitwise_and(image, image, mask=mask)
	image[:, :, 1] = frame[:, :, 1]
	image[:, :, 2] = frame[:, :, 2]

	image[:max_y, start_x:max_x] = cv2.addWeighted(draw_tool, 0.7, image[:max_y, start_x:max_x], 0.3, 0)

	cv2.putText(image, curr_tool, (270+start_x,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)#Text to pe put in the frame
	cv2.imshow("frame",image)
	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break
 