This is a pipeline I put together to get images to use transfer learning using a YOLOv8 model. Each module uses the command line with an intuitive selection system.  It consists of the following parts:

  a. image_extraction.py - Takes images or videos and prepares them for annotation. Videos are split up into a user-defined number of frames and saved as individual images

  b. annotation_tool.py - Simple GUI that lets the user build bounding boxes around whatever they want to classify

  c. yolo_conversion.py - Converts the newly annotated files into a format compatible with YOLO

  d. data_augmentation.py - Applies standard image preprocessing to the dataset

  e. dataset_splitting.py - Splits the data into training and validation sets while taking steps to prevent data leakage
