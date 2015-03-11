project( SensorFusion ) 
cmake_minimum_required(VERSION 2.8) 

#OPTION(BUILD_EXAMPLES "Build Examples" ON)


# Add to module path, so we can find our cmake modules 
set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules ${CMAKE_MODULE_PATH} )

#the boost libraries that we need
set(BOOST_LIBS thread date_time system timer)

find_package( CeresSolver REQUIRED)
find_package(Sophus REQUIRED)
find_package( Eigen3 REQUIRED )
find_package( MVL REQUIRED )
find_package( CVARS REQUIRED )
find_package( Boost COMPONENTS ${BOOST_LIBS} REQUIRED)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -Wall")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

set( _LIBRARIES ${MVL_LIBRARIES}
                ${BULLET_LIBRARIES}
                ${CeresSolver_LIBRARIES}
                ${BOOST_LIBRARIES}
                ${VRPN_LIBRARY}
#                ${Sophus_LIBRARIES}  )
                )
set( _INCLUDE_DIRS
        ${Sophus_INCLUDE_DIR}
        ${CeresSolver_INCLUDE_DIRS}
        ${CMAKE_BINARY_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${MVL_INCLUDE_DIRS}
        ${CVARS_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        )

link_directories(${Sophus_LIBRARY_DIRS})

# expose these for third parties
set( SENSORFUSION_INCLUDE_DIRS ${_INCLUDE_DIRS}  CACHE INTERNAL "" FORCE )
set( SENSORFUSION_LIBRARIES  ${_LIBRARIES} CACHE INTERNAL "" FORCE )
                           
include_directories( ${SENSORFUSION_INCLUDE_DIRS} )

set( SRCS
        SensorFusionCeres.cpp
        SensorFusionVo.cpp
   )

set( HDRS
        CostFunctions.h
        SensorFusionCeres.h
        SensorFusionVo.h
   )


add_library( SensorFusion ${SRCS} ${HDRS} )
target_link_libraries( SensorFusion ${SENSORFUSION_LIBRARIES} )

#IF(BUILD_EXAMPLES)
#    FIND_PACKAGE(SceneGraph REQUIRED)
#    INCLUDE_DIRECTORIES(${SceneGraph_INCLUDE_DIRS})

#    FIND_PACKAGE(Pangolin REQUIRED)
#    INCLUDE_DIRECTORIES(${Pangolin_INCLUDE_DIRS})
#    LINK_DIRECTORIES(${Pangolin_LIBRARY_DIRS})

#    add_executable( FusionTest FusionTest.cpp )
#    target_link_libraries( FusionTest ${SENSORFUSION_LIBRARIES} ${SceneGraph_LIBRARIES} ${Pangolin_LIBRARIES} SensorFusion )
#ENDIF()

get_target_property( _LIBRARY SensorFusion LOCATION )
list( INSERT SENSORFUSION_LIBRARIES 0 ${_LIBRARY} )
 
# Create the *Config.cmake file for the build tree.
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/SensorFusionConfig.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/SensorFusionConfig.cmake @ONLY IMMEDIATE )

# Add module to CMake package registery.
export( PACKAGE SensorFusion )

