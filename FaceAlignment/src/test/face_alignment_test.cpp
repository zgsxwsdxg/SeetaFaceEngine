/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face alignment, the
 * face alignment method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment,
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "cv.h"
#include "highgui.h"

#include "face_detection.h"
#include "face_alignment.h"

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
#endif
using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 4) {
    cout << "Usage: " << argv[0] <<
         "face_detection_model face_alignment_model image_file" << endl;
    return -1;
  }

  // Initialize face detection model
  seeta::FaceDetection
  detector(argv[1]);
  detector.SetMinFaceSize(40);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(2, 2);

  // Initialize face alignment model
  seeta::FaceAlignment point_detector(argv[2]);

  //load image
  IplImage *img_grayscale = NULL;
  img_grayscale = cvLoadImage(argv[3], 0);

  if (img_grayscale == NULL) {
    return 0;
  }

  IplImage *img_color = cvLoadImage(argv[3], 1);
  int pts_num = 5;
  int im_width = img_grayscale->width;
  int im_height = img_grayscale->height;
  unsigned char *data = new unsigned char[im_width * im_height];
  unsigned char *data_ptr = data;
  unsigned char *image_data_ptr = (unsigned char *)img_grayscale->imageData;
  int h = 0;

  for (h = 0; h < im_height; h++) {
    memcpy(data_ptr, image_data_ptr, im_width);
    data_ptr += im_width;
    image_data_ptr += img_grayscale->widthStep;
  }

  seeta::ImageData image_data;
  image_data.data = data;
  image_data.width = im_width;
  image_data.height = im_height;
  image_data.num_channels = 1;

  // Detect faces
  std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
  int32_t face_num = static_cast<int32_t>(faces.size());

  if (face_num == 0) {
    delete[]data;
    cvReleaseImage(&img_grayscale);
    cvReleaseImage(&img_color);
    return 0;
  }

  cout << "Face Num is " << face_num << endl;
  for (int i = 0; i < face_num; ++i) {
    // Detect 5 facial landmarks
    seeta::FacialLandmark points[5];
    point_detector.PointDetectLandmarks(image_data, faces[i], points);
    // Visualize the results
    cvRectangle(img_color, cvPoint(faces[i].bbox.x, faces[i].bbox.y),
                cvPoint(faces[i].bbox.x + faces[i].bbox.width - 1,
                        faces[i].bbox.y + faces[i].bbox.height - 1), CV_RGB(255, 0, 0));

    for (int j = 0; j < pts_num; j++) {
      cvCircle(img_color, cvPoint(points[j].x, points[j].y), 2, CV_RGB(0, 255, 0),
               CV_FILLED);
    }
  }

  string save_path = argv[3];
  save_path += "_alignment.jpg";
  cvSaveImage(save_path.c_str(), img_color);

  // Release memory
  cvReleaseImage(&img_color);
  cvReleaseImage(&img_grayscale);
  delete[]data;
  data = NULL;
  return 0;
}
