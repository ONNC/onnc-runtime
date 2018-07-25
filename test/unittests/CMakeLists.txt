function(operator_unittest arg)
    add_executable(test_operator_${arg}
        test_operator_${arg}.cpp
    )
    add_dependencies(test_operator_${arg}
        Ext_SkyPat
    )
    target_link_libraries(test_operator_${arg}
        onnc-runtime
        skypat
    )
add_test(Operator_${arg} test_operator_${arg})
endfunction(operator_unittest)

operator_unittest("and")
operator_unittest("add")
operator_unittest("acos")
operator_unittest("asin")
operator_unittest("abs")
operator_unittest("ceil")
operator_unittest("log")
operator_unittest("equal")
operator_unittest("cos")
operator_unittest("argmax")
operator_unittest("argmin")
operator_unittest("upsample")
operator_unittest("sin")
operator_unittest("mean")
operator_unittest("reshape")
operator_unittest("pow")
operator_unittest("atan")
operator_unittest("flatten")
operator_unittest("leakyrelu")
operator_unittest("exp")
operator_unittest("sigmoid")
operator_unittest("clip")
operator_unittest("affine")
operator_unittest("hardsigmoid")
