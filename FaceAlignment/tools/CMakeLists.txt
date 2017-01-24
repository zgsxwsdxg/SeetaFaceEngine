file(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(depend_libs opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs seeta_facedet_lib seeta_fa_lib)

foreach(source ${src_files})
  get_filename_component(file_name ${source} NAME_WE)
  set(tool_name "${file_name}.bin")
  # target
  add_executable(${tool_name} ${source})
  target_link_libraries(${tool_name} ${depend_libs})
endforeach(source)
