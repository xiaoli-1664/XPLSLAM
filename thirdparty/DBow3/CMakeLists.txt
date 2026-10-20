cmake_minimum_required(VERSION 3.24)
project(DBoW3 LANGUAGES C CXX)
find_package(OpenCV 4 REQUIRED)
add_library(DBoW3 SHARED
    src/BowVector.cpp src/Database.cpp src/DescManip.cpp
    src/FeatureVector.cpp src/QueryResults.cpp src/ScoringObject.cpp
    src/Vocabulary.cpp src/quicklz.c)
target_compile_features(DBoW3 PUBLIC cxx_std_17)
target_compile_definitions(DBoW3 PRIVATE DBOW_DSO_EXPORTS OPENCV_VERSION_3)
target_include_directories(DBoW3 PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DBoW3 PUBLIC ${OpenCV_LIBS})
