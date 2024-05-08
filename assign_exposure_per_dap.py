import csv
from datetime import datetime
import glob
import os
import re
import cv2
import numpy as np
from alive_progress import alive_bar
from time import strftime, localtime
from scipy.interpolate import Rbf
import ast
from ultralytics import YOLO

from doctor_class import Doctor


# ===> UTIL METHODS
def calculate_occupational_exposure(placeholder_var):
     return placeholder_var * 10


def read_filenames_from_folder(folder_name, file_extension):
      filenames = glob.glob(os.path.join(folder_name, file_extension))
      filenames.sort()
      return filenames


def create_interpolation():
     # This is the data taken from Jordan Brinkerhoff's masters thesis:
      x = np.array([0, 0, 0, 0.35, 0.71, 1.06, 1.15, 1.81, 2.34, 2.34, 2.34, 1.41, -0.35, -0.71, -1.06, -1.15, -1.81, -2.34, -2.34, -2.34, -1.41, -6, 6, -6, 6, 4.24, -4.24, 2.12, -2.12, 0, 0, 0, 0, 0])
      y = np.array([-1, -0.5, -1.5, -0.35, -0.71, -1.06, -0.5, -0.5, -0.5, -1, -1.5, -1.41, -0.35, -0.71, -1.06, -0.5, -0.5, -0.5, -1, -1.5, -1.41, 0, 0, -6, -6, -4.24, -4.24, -2.12, -2.12, -4.24, -2.12, 0, -6, -3])
      values = np.array([35.2, 47.6, 23.8, 57.1, 33.3, 12.8, 28.6, 15.2, 9.5, 9.5, 8.57, 9.86, 57.1, 33.3, 12.8, 28.6, 15.2, 9.5, 9.5, 8.57, 9.86, 0.01, 0.01, 0.01, 0.01, 0.12, 0.12, 0.48, 0.48, 0.12, 0.48, 65, 0.12, 0.48])

      # Perform interpolation
      rbf = Rbf(x, y, values, function='linear')

      # Return object to be used for measuring scatter radiation at a certain position
      return rbf


# ===> LOADING DEEPSORT
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)
unique_track_ids = set()


# ===> INITIALISATION FOR REALSENSE FILES
folder = "intel_realsense_data_folder"

print(" - Reading image filenames from folder")
images = read_filenames_from_folder(folder, "*.png")

print(" - Reading csv filenames from folder")
csv_files = read_filenames_from_folder(folder, "*.csv")

print(" - Reading txt filenames from folder")
txt_files = read_filenames_from_folder(folder, "*.txt")


# ===> PROCESS EACH FRAME FROM REALSENSE DATA
print(" - Loading in YoloV8 model")
"""
Ensure the model is downloaded locally for this script to function. The code 
is currently modified for YoloV8. It is possible to change to other versions,
however, code modification will be required.
"""
model = YOLO("yolov8n.pt")  # this can be changed to include different V8 models

print(" - Creating CSV file to keep track of data extracted from RealSense")
with open("realsense_data_output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp (24hr)", "Timestamp (epoch)", "No. of Doctors", "Depths (list)", "X-Positions (list)", "Tracking IDs"])


"""
This Python program can be split into two main sections. The first section 
deals with iterating through the data gathered by the Intel RealSense, and the
second section deals with iterating through the data gathered by the DoseWise 
program and synching the data across the two sources. This is the beginning of
the first section.
"""

print(" - Iterating through list of loaded images")
with alive_bar(len(images)) as bar:
      for i in range(0, len(images)):
            images[i] = cv2.imread(images[i])  # load the image for processing

            results = model(images[i], conf=0.8, verbose=False)  # run inference

            # Data to extract from the frame:
            number_of_doctors = 0
            depths_of_doctors = []
            x_position_of_doctors = []

            # This section scans for objects:    
            for result in results:
                  boxes = result.boxes
                  cls = boxes.cls.tolist()  # Convert tensor to list
                  xyxy = boxes.xyxy
                  conf = boxes.conf
                  xywh = boxes.xywh  # box with xywh format, (N, 4)
            
            # This section deals with tracking objects and assigning them IDs:
            pred_cls = np.array(cls)
            conf = conf.detach().cpu().numpy()
            xyxy = xyxy.detach().cpu().numpy()
            bboxes_xywh = xywh
            bboxes_xywh = xywh.cpu().numpy()
            bboxes_xywh = np.array(bboxes_xywh, dtype=float)

            tracks = tracker.update(bboxes_xywh, conf, images[i])

            list_of_ids_for_frame = []

            tracked_objects = tracker.tracker.tracks

            for track in tracked_objects:
                  track_id = track.track_id
                  list_of_ids_for_frame.append(track_id)
                  hits = track.hits
                  x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates
                  w = x2 - x1  # Calculate width
                  h = y2 - y1  # Calculate height

                  number_of_doctors += 1

                  x_position = (int(x1) + int(x2)) / 2  # average x-position
                  x_position_of_doctors.append(x_position)

                  """
                  Since the depth and colour have different resolutions scaling is
                  required the width and height is to be divided by exactly 1.5 
                  """
                  x1_depth = int(int(x1) / 1.5)
                  x2_depth = int(int(x2) / 1.5)
                  y1_depth = int(int(y1) / 1.5)
                  y2_depth = int(int(y2) / 1.5)

                  # prevents out of range indexes for pixel values
                  # (sometimes the value can be 848 or 480 causing an error)
                  x2_depth = min(x2_depth, 847)
                  y2_depth = min(y2_depth, 479)

                  # Calculate the distance to the object
                  object_depth_total = 0

                  # Used to collect all rows from bounding box into one row
                  long_row_of_distances = []

                  list_of_rows_entire_csv = []
                  with open(csv_files[i]) as file_obj: 
                        # Create reader object by passing the file object to reader method
                        reader_obj = csv.reader(file_obj) 
                        for row in reader_obj:
                              list_of_rows_entire_csv.append(row)

                  for j in range(int(y1_depth), int(y2_depth)):
                        try:
                              row_bounding_box_values = [eval(i) for i in (list_of_rows_entire_csv[j][x1_depth: x2_depth])]
                              long_row_of_distances.append(row_bounding_box_values)
                        except IndexError:
                              print("Index error")
                              print(x1_depth)
                              print(x2_depth)
                              print(y1_depth)
                              print(y2_depth)
                              print(csv_files[i])                           
                              print(list_of_rows_entire_csv[j])

                  # median of all distances inside bounding box
                  object_depth = np.median(long_row_of_distances)

                  # create label for depth
                  label = f"{object_depth:.2f}m"
                  
                  # append depth to array
                  depths_of_doctors.append(object_depth)

                  # annotate image with bounding box and text box
                  cv2.rectangle(images[i], (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)  # red box
                  text_color = (255, 255, 255)  # Black color for text
                  cv2.rectangle(images[i], (int(x1), int(y1-40)), (int(x1 + w), int(y1)), (0,0,0), -1)
                  
                  # annotating image with person ID
                  cv2.putText(images[i], f"Person: {track_id}", (int(x1), int(y1) - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
                  
                  # annotating image with person X and Y position
                  x_position_m_label = round((float(x_position) * 0.004) - 1.34, 2)
                  y_position_m_label = round(2.5 - float(object_depth), 2)
                  cv2.putText(images[i], f"X: {x_position_m_label}m",(int(x1 + 5), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
                  cv2.putText(images[i], f"Y: {y_position_m_label}m", (int((x1 + x2) / 2), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
                  
                  # Add the track_id to the set of unique track IDs
                  unique_track_ids.add(track_id)


            # This section deals with extracting the timestamp from the txt files
            with open(txt_files[i], 'r') as file:
                  content = file.read()
                  match = re.search(r"Time Of Arrival: (\d+)", content)
                  if match:
                        timestamp_epoch = int(match.group(1))
                  else:
                        print("Error: 'Time Of Arrival' not found in this file.")
                  
                  timestamp_epoch = timestamp_epoch // 10 // 10 // 10
                  timestamp = strftime('%H:%M:%S', localtime(timestamp_epoch))

            # The list contains information in the order of tracked IDs. 
            # (Person 1 will always be index 0 and Person 2 index 1, etc.)
            list_for_csv = [timestamp, timestamp_epoch, number_of_doctors, 
                            depths_of_doctors, x_position_of_doctors, 
                            list_of_ids_for_frame]

            # Save extracted information to a CSV file
            with open("realsense_data_output.csv", "a", newline="") as file:
                  writer = csv.writer(file)
                  writer.writerow(list_for_csv)

            bar()


# ===> OUTPUTTING VIDEO FILE:
print(" - Initialising video writer")
output_video_name = "resulting_video.mp4"
codec = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_name, codec, fps=30, frameSize=(1280, 720))

# Create a video of the annotated images with bounding boxes
for j in range(len(images)):
      video.write(images[j])

print(" - Closing cv2 windows")
cv2.destroyAllWindows()


"""
Again, this Python program can be split into two main sections. The second 
section deals with iterating through the data gathered by the DoseWise program
and synching the data across the two sources. This is the end of the first 
section and the beginning of the second section.
"""

# ===> PROCESS DOSEWISE CSV

print(" - Creating CSV file to keep track of occupational exposure")
with open("occupational_exposure_output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([])

print(" - Create interpolation map object")
interpolated_values = create_interpolation()

dosewise_filename = "dosewise_data.csv"

# used to keep track of individuals and their occupational exposure
doctor_objects = []

# list for doctor names - this can be improved to create a dynamic system for naming
doctor_names = ["Sebastian Fitzgerald", "Edward Griffin", "Saoirse Kavanagh", 
                "Adam Lyons", "Jacob Brown", "John MacLaughlin", 
                "Hazel Casey", "Leo Keogh", "Mary Mullan", "Erin Boyle" ]

# Iterate through each line of the DoseWise CSV file
with open(dosewise_filename, newline='') as csvfile:

      reader = csv.DictReader(csvfile)
      list_values = list(reader)

      # count how many data points are in one minute of data
      data_points_for_minute = 0

      # list of dap values within the minute
      dap_values_for_minute = []

      # The date is in MM/DD/YYYY format eg. '12/13/2023'
      acquisition_date = list_values[1]['Acquisition start date']
      date_elements = acquisition_date.split('/')
      date_year = date_elements[2]
      date_month = date_elements[0]
      date_day = date_elements[1]

      # Iterate through a minute of data points
      for i in range(0, len(list_values) - 1):
            current_time = list_values[i]['Acquisition start time']

            if not current_time == "":
                  current_in_time = datetime.strptime(current_time, "%I:%M %p")
                  current_hours = datetime.strftime(current_in_time, "%H")
                  current_minutes = datetime.strftime(current_in_time, "%M")

                  next_time = list_values[i+1]['Acquisition start time']
                  next_in_time = datetime.strptime(next_time, "%I:%M %p")
                  next_hours = datetime.strftime(next_in_time, "%H")
                  next_minutes = datetime.strftime(next_in_time, "%M")

                  # convert times to epoch format
                  current_time = int(datetime(int(date_year), int(date_month), int(date_day), int(current_hours), int(current_minutes), 0).timestamp())
                  next_time = int(datetime(int(date_year), int(date_month), int(date_day), int(next_hours), int(next_minutes), 0).timestamp())

            data_points_for_minute += 1

            dap_values_for_minute.append(float(list_values[i]['DAP']))

            # If loop comes to the next minute process the current minute:
            if not current_time == next_time:

                  # Divide the minute by the number of DoseWise datapoints for current minute
                  minute_division = 60 // data_points_for_minute
                  number_of_RL_pts_in_section = 30 * minute_division  # 30fps * time period (seconds)


                  # Used to avoid parsing entire file for each DoseWise minute section datapoint
                  loop_begin_index = 0

                  for point in range(0, data_points_for_minute):

                        #Set start and end times for the minute division
                        start_time = current_time + (minute_division * point)
                        end_time = current_time +(minute_division * (point + 1))

                        # Open up the realsense data CSV for position between certain timestamps.
                        with open("realsense_data_output.csv", newline='') as csvfile:
                              
                              reader = csv.DictReader(csvfile)
                              oe_values = list(reader)
                              first_point_is_found = False

                              # This variable can be improved to dynamically read the number of doctors.
                              # The two data sources won't start at the same time, so the first value
                              # needs to be synced up.
                              previous_num_of_doctors = int(oe_values[242]['No. of Doctors'])  # initialise variable

                              # Loop through entire RealSense data CSV file.
                              for j in range(loop_begin_index, len(oe_values) - 1):
                                    
                                    # Check whether at the start or end between two timestamps.                            
                                    rs_time_epoch = int(oe_values[j]['Timestamp (epoch)'])

                                    if rs_time_epoch == start_time and not first_point_is_found:
                                          first_point_is_found = True

                                    elif rs_time_epoch == end_time:     # This will end the loop
                                          first_point_is_found = False
                                          loop_begin_index = j          # Avoid looping from beginning again
                                          break

                                    # If the first point has been found start parsing the corresponding data for the timestamp.
                                    if first_point_is_found:
                                          rs_no_doctors = int(oe_values[j]['No. of Doctors'])
                                          rs_depths_list = ast.literal_eval(oe_values[j]['Depths (list)'])
                                          rs_positions_list = ast.literal_eval(oe_values[j]['X-Positions (list)'])
                                          rs_person_ids = ast.literal_eval(oe_values[j]['Tracking IDs'])

                                          # Keep track of which individuals are present
                                          list_of_doctor_ids = []
                                          for doctor in doctor_objects:
                                                list_of_doctor_ids.append(doctor.id)

                                          # Keep track of occupational exposure for each individual
                                          list_of_oe_for_point = []

                                          # Assign radiation exposure values
                                          for index, id in enumerate(rs_person_ids):

                                                x_position_metres = (float(rs_positions_list[index]) * 0.004) - 1.34
                                                y_position_metres = 2.5 - float(rs_depths_list[index])
                                          
                                                scatter_radiation = interpolated_values(x_position_metres, y_position_metres)

                                                occupational_exposure = scatter_radiation * (dap_values_for_minute[point] / number_of_RL_pts_in_section)

                                                list_of_oe_for_point.append(occupational_exposure)

                                                # If the doctor is present append the calculated occupational exposure
                                                if id in list_of_doctor_ids:
                                                      for doctor in doctor_objects:
                                                            if doctor.id == id:
                                                                  doctor.add_occupational_exposure(occupational_exposure)
                                                
                                                # Otherwise create a new doctor object
                                                else:
                                                      # For ease of use, the doctors are given names from a list, but for real-world use the doctors can be given names dynamically
                                                      doctor_objects.append(Doctor(doctor_names[id], id))

                                                      """
                                                      # To dynamically add a doctor:
                                                      doctor_name = input("New individual detected, what is their name? ")
                                                      doctor_objects.append(Doctor(doctor_name, id))
                                                      """

                                                      # Append OE
                                                      for doctor in doctor_objects:
                                                            if doctor.id == id:
                                                                  doctor.add_occupational_exposure(occupational_exposure)


                                          # Output the OE values to a csv file (not required for program but useful for checking data points)
                                          with open('occupational_exposure_output.csv', 'a', newline="") as file:
                                                writer = csv.writer(file)
                                                writer.writerow(list_of_oe_for_point)

                  print("\n=> Resulting Occupational Exposure from Surgery:")
                  for doctor in doctor_objects:
                        print(doctor)
                  
                  print("")

                  data_points_for_minute = 0  # reset counter

                  # loop back to process the next minute from the DoseWise CSV file
