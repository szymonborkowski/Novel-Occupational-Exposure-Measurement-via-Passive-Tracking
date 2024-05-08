# Novel Occupational Exposure Measurement via Passive Tracking
This repository contains the Python script as part of my master's thesis project.

If you are interested in seeing the process and results of the project you can read the master's thesis.

If you are interested in aspects of the project not uploaded to the repository please get in contact with me using the email (szymonborkowski10@gmail.com).

# Project Abstract
Fluorography is used in medical procedures like angioplasty to visualise arteries and veins in real-time using X-rays. The nature of these surgeries means that the cardiologists performing the procedure need to be present in the surgery room while an X-ray takes place, which results in high occupational radiation exposure during these procedures due to scatter radiation. Despite efforts to mitigate risks, some staff do not comply with safety measures such as utilising lead shields. Observational data suggests compliance issues with wearing dosimeter badges which are meant to measure radiation exposure. 

To address the compliance issues, this project utilises an Intel RealSense D455 camera to monitor each cardiologist’s position throughout procedures in the surgery room by utilising the camera’s feature to measure distance. Previously done research measured scatter radiation at different positions in the surgery room which can now be utilised to create a map of radiation at all positions. The DoseWise program present on the X-ray outputs information for determining how scatter radiation changes such as X-ray position and intensity including timestamps at which the X-rays take place. Individuals will be tracked using object tracking and a manual labelling system assisted by the radiographers present in the surgery control room.

Using object tracking and distance measurement the proposed method tracks the location of cardiologists relative to the radiation source and utilises this information to calculate the radiation dose experienced by each cardiologist during surgery.
In hospitals around Europe the issue of measuring cardiologists’ occupational exposure exists. This proposed solution provides a non-invasive method of tracking exposure, reducing reliance on staff compliance, and improving safety in hospitals. The goal of the project is to prove individuals can be tracked over time and creating a proof-of-concept occupational exposure measurement system.

# Acknowledgements
Thank you to Peter Conneely, Emerald House, Keith Scully, Brendan Tuohy, from Galway University Hospital, and Dr. Brian Deegan from University of Galway for all the help throughout the project. Thank you to Dr. Darragh Mullins for providing me with the required equipment to complete the project.