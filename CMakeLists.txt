project( SensorFusion )
cmake_minimum_required(VERSION 2.8)

# Add to module path, so we can find our cmake modules
set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules ${CMAKE_MODULE_PATH} )

STRING( TOLOWER ${PROJECT_NAME} LIBRARY_NAME )

include(install_package)

#the boost libraries that we need
#set(BOOST_LIBS thread date_time system timer)

find_package( CeresSolver REQUIRED)
find_package( Sophus REQUIRED)
find_package( Eigen3 REQUIRED )
find_package( CVars REQUIRED )
find_package( CarPlanner REQUIRED )
#find_package( Boost COMPONENTS ${BOOST_LIBS} REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -Wall")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

set( _LIBRARIES
                ${BULLET_LIBRARIES}
                ${CeresSolver_LIBRARIES}
                #                ${BOOST_LIBRARIES}
                ${VRPN_LIBRARY}
                ${CVars_LIBRARIES}
                ${CarPlanner_LIBRARIES} )

set( _INCLUDE_DIRS
    ${Sophus_INCLUDE_DIRS}
    ${CeresSolver_INCLUDE_DIRS}
        ${CMAKE_BINARY_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${CVars_INCLUDE_DIRS}
        ${CarPlanner_INCLUDE_DIRS}
        )

# expose these for third parties
set( SensorFusion_INCLUDE_DIRS ${_INCLUDE_DIRS}  CACHE INTERNAL "" FORCE )
set( SensorFusion_LIBRARIES  ${_LIBRARIES} CACHE INTERNAL "" FORCE )

include_directories( ${SensorFusion_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include )

set( SRCS
        SensorFusionCeres.cpp
        SensorFusionVo.cpp
   )

set( HDR_PREFIX "include/${PROJECT_NAME}" )
set( HDRS
    ${HDR_PREFIX}/CostFunctions.h
        ${HDR_PREFIX}/SensorFusionCeres.h
        ${HDR_PREFIX}/SensorFusionVo.h
   )


add_library( sensorfusion SHARED ${SRCS} ${HDRS} )
target_link_libraries( sensorfusion ${SensorFusion_LIBRARIES} )

install_package(
    PKG_NAME ${PROJECT_NAME}
    LIB_NAME ${LIBRARY_NAME}
    VERSION 2.1
    DESCRIPTION "Simple sensor fusion library for IMU and process model."
    INSTALL_INCLUDE_DIR true
    INSTALL_GENERATED_HEADERS ${GENERATED_HEADERS}
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    INCLUDE_DIRS ${SensorFusion_INCLUDE_DIRS}
    LINK_LIBS ${SensorFusion_LIBRARIES}
    )

###########################

include(${CMAKE_MODULE_PATH}/cmake_uninstall.cmake.in)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
###########################
