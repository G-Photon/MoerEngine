set(target_name TestMath)

add_executable(${target_name} main.cpp)
target_link_libraries(${target_name} PUBLIC Moer::Core)

set_target_properties(${target_name} PROPERTIES FOLDER ${moer_test_folder})

add_executable(TestVectorNormalize NormalizeTest.cpp)
target_link_libraries(TestVectorNormalize PRIVATE Moer::Core)
set_target_properties(TestVectorNormalize PROPERTIES FOLDER ${moer_test_folder})
add_test(NAME VectorNormalize COMMAND $<TARGET_FILE:TestVectorNormalize>)
set_tests_properties(VectorNormalize PROPERTIES TIMEOUT 15)
