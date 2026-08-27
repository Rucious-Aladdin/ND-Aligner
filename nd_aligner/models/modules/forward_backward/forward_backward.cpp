#include <torch/extension.h>
#include <c10/util/Optional.h>


torch::Tensor log_alpha_forward_cuda(
    torch::Tensor log_b,
    torch::Tensor mask,
    c10::optional<torch::Tensor> opt_sep_mask,
    double neg_large
);

torch::Tensor log_beta_forward_cuda(
    torch::Tensor log_b,
    torch::Tensor mask,
    c10::optional<torch::Tensor> opt_sep_mask,
    double neg_large
);


torch::Tensor log_alpha_backward_cuda(
    torch::Tensor grad_log_alpha,
    torch::Tensor log_b,
    torch::Tensor log_alpha,
    torch::Tensor mask,
    c10::optional<torch::Tensor> opt_sep_mask,
    double neg_large
);

torch::Tensor log_beta_backward_cuda(
    torch::Tensor grad_log_beta,
    torch::Tensor log_b,
    torch::Tensor log_beta,
    torch::Tensor mask,
    c10::optional<torch::Tensor> opt_sep_mask,
    double neg_large
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "log_alpha_forward",
        &log_alpha_forward_cuda,
        "Monotone CRF log-alpha forward (CUDA)",
        pybind11::arg("log_b"),
        pybind11::arg("mask"),
        pybind11::arg("opt_sep_mask") = c10::nullopt,
        pybind11::arg("neg_large") = -1.0e9
    );

    module.def(
        "log_beta_forward",
        &log_beta_forward_cuda,
        "Monotone CRF log-beta forward (CUDA)",
        pybind11::arg("log_b"),
        pybind11::arg("mask"),
        pybind11::arg("opt_sep_mask") = c10::nullopt,
        pybind11::arg("neg_large") = -1.0e9
    );

    module.def(
        "log_alpha_backward",
        &log_alpha_backward_cuda,
        "Monotone CRF log-alpha backward (CUDA)",
        pybind11::arg("grad_log_alpha"),
        pybind11::arg("log_b"),
        pybind11::arg("log_alpha"),
        pybind11::arg("mask"),
        pybind11::arg("opt_sep_mask") = c10::nullopt,
        pybind11::arg("neg_large") = -1.0e9
    );

    module.def(
        "log_beta_backward",
        &log_beta_backward_cuda,
        "Monotone CRF log-beta backward (CUDA)",
        pybind11::arg("grad_log_beta"),
        pybind11::arg("log_b"),
        pybind11::arg("log_beta"),
        pybind11::arg("mask"),
        pybind11::arg("opt_sep_mask") = c10::nullopt,
        pybind11::arg("neg_large") = -1.0e9
    );
}