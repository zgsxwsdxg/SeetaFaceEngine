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

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "face_detection.h"
#include "face_alignment.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "Usage: " << argv[0] <<
         " face_detection_model face_alignment_model" << endl;
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

// read images lists file

  string datas_dir = "/home/dxg/work_place/SeetaFaceEngine/FaceAlignment/data";
  string image_format = ".jpg";
  cout << "Data directory:" << datas_dir << endl;
  string images_dir = datas_dir + "/images/";
  string results_dir = datas_dir + "/results/";
  string results_images_dir = results_dir + "images/";
  double average_time = 0.0f;
  int image_total = 0;

  for (int list_idx = 1; list_idx <= 10; ++list_idx) {
    char fddb[300];
    char fddb_out[300];
    char fddb_answer[300];
    sprintf(fddb, "%s/FDDB-folds/FDDB-fold-%02d.txt", datas_dir.c_str(), list_idx);
    sprintf(fddb_out, "%s/results/fold-%02d-out.txt", datas_dir.c_str(), list_idx);
    sprintf(fddb_answer, "%s/FDDB-folds/FDDB-fold-%02d-ellipseList.txt",
            datas_dir.c_str(),
            list_idx);
    cout << "FDDB list file:" << fddb << endl;
    std::ifstream file_reader;
    file_reader.open(fddb, std::ios::in);

    if (!file_reader.is_open()) {
      cout << "Open image list file failed.\n";
      continue;
    }

    FILE *fanswer = fopen(fddb_answer, "r");

    if (NULL == fanswer) {
      cout << "Open image annotation failed.\n";
      cout << "Please check file:" << fddb_answer << endl;
      break;
    }

    FILE *fout = fopen(fddb_out, "w");

    if (NULL == fout) {
      cout << "Open result output file failed.\n";
      break;
    }

    int images_cnt = 0;

    while (!file_reader.eof()) {
      string row_text = "";
      getline(file_reader, row_text, '\n');

      if (row_text.empty()) {
        continue;
      }

      string full_path = images_dir + row_text;
      full_path += image_format;
      //load image
      cv::Mat gray_image, color_image;
      gray_image = imread(full_path, IMREAD_GRAYSCALE);

      if (gray_image.empty()) {
        continue;
      }

      images_cnt++;
      image_total++;
      color_image = imread(full_path.c_str(), IMREAD_COLOR);

      int pts_num = 5;
      int im_width = gray_image.cols;
      int im_height = gray_image.rows;

      unsigned char *data = new unsigned char[im_width * im_height];

      if (NULL == data) {
        images_cnt--;
        continue;
      }

      unsigned char *data_ptr = data;
      unsigned char *image_data_ptr = (unsigned char *)(gray_image.data);
      int h = 0;

      for (h = 0; h < im_height; h++) {
        memcpy(data_ptr, image_data_ptr, im_width);
        data_ptr += im_width;
        image_data_ptr += gray_image.step[0];
      }

      seeta::ImageData image_data;
      image_data.data = data;
      image_data.width = im_width;
      image_data.height = im_height;
      image_data.num_channels = 1;

      // Detect faces
      long t0 = cv::getTickCount();
      std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
      int32_t face_num = static_cast<int32_t>(faces.size());
      long t1 = cv::getTickCount();
      double secs = (t1 - t0) / cv::getTickFrequency();
      average_time += secs;
      cout << "fold:" << list_idx << ",image path:" << row_text << ",time: " << secs
           << " seconds " << endl;

      fprintf(fout, "%s\n%d\n", row_text.c_str(), face_num);

      for (int i = 0; i < face_num; ++i) {
        // Detect 5 facial landmarks
        seeta::FacialLandmark points[5];
        point_detector.PointDetectLandmarks(image_data, faces[i], points);
        // Visualize the results
        cv::Rect face_roi;
        face_roi.x = faces[i].bbox.x;
        face_roi.y = faces[i].bbox.y;
        face_roi.width = faces[i].bbox.width - 1;
        face_roi.height = faces[i].bbox.height - 1;
        fprintf(fout, "%d %d %d %d %lf\n", face_roi.x, face_roi.y, face_roi.width,
                face_roi.height, faces[i].score);
        char str_score[30];
        sprintf(str_score, "%.4lf", faces[i].score);
        // face roi
        cv::rectangle(color_image, face_roi, cv::Scalar(255, 0, 0), 3);
        // face score
        cv::putText(color_image, str_score, cv::Point(face_roi.x, face_roi.y),
                    cv::FONT_HERSHEY_PLAIN, 1,
                    cv::Scalar(0, 255, 0), 2);

        // facial landmarks
        for (int j = 0; j < pts_num; j++) {
          cv::circle(color_image, cv::Point(points[j].x, points[j].y), 3, cv::Scalar(0,
                     255, 0), -1);
        }
      }

      // draw answer face
      int an_face_num = 0;
      char ans_image_path[400];
      fscanf(fanswer, "%s", ans_image_path);
      fscanf(fanswer, "%d", &an_face_num);

      for (int k = 0; k < an_face_num; k++) {
        double major_axis_radius, minor_axis_radius, angle, center_x, center_y, score;
        fscanf(fanswer, "%lf %lf %lf %lf %lf %lf", &major_axis_radius,
               &minor_axis_radius, \
               &angle, &center_x, &center_y, &score);
        // draw answer
        angle = angle / 3.1415926 * 180.;
        cv::ellipse(color_image, Point2d(center_x, center_y), Size(major_axis_radius,
                    minor_axis_radius), \
                    angle, 0., 360., Scalar(0, 0, 255), 2);
      }

      string save_path = results_images_dir;
      string image_name = cv::format("%02d_%03d_%02d.jpg", list_idx, images_cnt,
                                     face_num);
      save_path += image_name;
//      cout << "image result path:" << save_path << endl;
      imwrite(save_path, color_image);

      delete[]data;
      data = NULL;
    }

    file_reader.close();
    fclose(fout);
    fclose(fanswer);
  }

  if (image_total > 0) {
    cout << "Detection average time:" << average_time / image_total <<
         "s,total images:" << image_total << endl;
  }

  return 0;
}
