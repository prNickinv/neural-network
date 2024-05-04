project(tests)

enable_testing()
add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(test-net TestActivationFunction.cpp)
target_link_libraries(test-net gtest gtest_main)
target_link_libraries(test-net net-lib)

include(GoogleTest)
gtest_discover_tests(test-net)
