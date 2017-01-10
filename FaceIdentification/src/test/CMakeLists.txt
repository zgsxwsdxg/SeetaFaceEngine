aux_source_directory (. SRC_LIST)

message(${SRC_LIST})

# add external libraries
#find_package(OpenCV REQUIRED)

set(depend_libs viplnet seeta_facedet_lib seeta_fa_lib opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)
enable_testing ()
foreach (f ${SRC_LIST})
  string(REGEX REPLACE "[.]cpp" ".bin" BIN ${f})
  add_executable(${BIN} ${f})
  target_link_libraries(${BIN} ${depend_libs})
endforeach ()

