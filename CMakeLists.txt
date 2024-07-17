#Copyright © 2024 Ryoichi Ishikawa All right reserved.
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)

# define a variable of project name
set( project_name "LC-calib_vc")
project (${project_name})


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


add_executable(LC-calib_vc
	src/direct_calib.cpp
	src/direct_calib/visualization.h
	src/direct_calib/visualization.cpp
	src/direct_calib/fusion.h
	src/direct_calib/fusion.cpp
	src/camera.cpp
	src/camera.h
	src/direct_calib/mesh_gen.h
	src/direct_calib/mesh_gen.cpp
)

add_executable(framebyframe_motionest
	src/frame_by_frame.cpp
	src/camera.cpp
	src/camera.h
	src/mb_calibration.cpp
	src/mb_calibration.h

)


add_executable(fusionba
	src/fusionBA.cpp
	src/SensorFusionBA.cpp
	src/SensorFusionBA.h
	src/BA_funcs.h
	src/fusionba/BasicPly.cpp
	src/fusionba/BasicPly.h
	src/fusionba/image_utility.cpp
	src/fusionba/image_utility.h
	src/fusionba/PointSequenceLoader.cpp
	src/fusionba/PointSequenceLoader.h
	src/fusionba/SensorFusion.cpp
	src/fusionba/SensorFusion.h
	src/fusionba/utility.cpp
	src/fusionba/utility.h
)

find_package(Eigen3 REQUIRED)
find_package(opengv REQUIRED)
if (opengv_FOUND)
    target_link_libraries(framebyframe_motionest opengv)
endif (opengv_FOUND)

find_package(ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
target_link_libraries(framebyframe_motionest ${CERES_LIBRARIES})
target_link_libraries(fusionba ${CERES_LIBRARIES})

find_package(nlohmann_json 3.2.0 REQUIRED)
target_link_libraries(LC-calib_vc nlohmann_json::nlohmann_json)
target_link_libraries(framebyframe_motionest nlohmann_json::nlohmann_json)
target_link_libraries(fusionba nlohmann_json::nlohmann_json)



find_package(cvl_toolkit REQUIRED)
target_link_libraries(framebyframe_motionest cvl_toolkit)
target_link_libraries(LC-calib_vc cvl_toolkit)
target_link_libraries(fusionba cvl_toolkit)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  include_directories(
      ${OpenCV_INCLUDE_DIRS}
    )
	target_link_libraries(LC-calib_vc
    ${OpenCV_LIBS}
    )
	target_link_libraries(framebyframe_motionest
    ${OpenCV_LIBS}
    )
	target_link_libraries(fusionba
		${OpenCV_LIBS}
    )
endif(OpenCV_FOUND)

find_package(camera_simulator REQUIRED)
target_link_libraries(LC-calib_vc camera_simulator)
find_package(GLEW REQUIRED)
if (GLEW_FOUND)
	message(STATUS "GLEW ${GLEW_INCLUDE_DIR}")
    target_link_libraries(LC-calib_vc  GLEW::GLEW)
endif()

find_package(dlib REQUIRED)
target_link_libraries(LC-calib_vc dlib::dlib)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(LC-calib_vc  ${PCL_LIBRARIES})


find_package( OpenMP )
if (OPENMP_FOUND) 
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
	src
	include
	${EIGEN3_INCLUDE_DIR}
)

