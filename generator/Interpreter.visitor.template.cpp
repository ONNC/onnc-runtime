#define restrict __restrict__
void ONNC_RUNTIME_${operator_name}_float(
  void * restrict onnc_runtime_context
  ${input_parameters_float}
  ${output_parameters_float}
  ${attribute_paprameters_float}
) {}

void Interpreter::visit${OperatorName}(::onnx::Node *pNode) {
  // Prepare input
  ${visitor_prepare_input}
  // Prepare output
  ${visitor_prepare_output}
  // Prepare attributes
  ${visitor_prepare_attribute}

  // Call to Runtime
  ONNC_RUNTIME_${operator_name}_float(
    m_pContext
    ${visitor_pass_input}
    ${visitor_pass_output}
    ${visitor_pass_attribute}
  );
};
