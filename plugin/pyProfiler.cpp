#include "trt_profiler.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(PyProfiler, m)
{
  namespace py = pybind11;

  py::module::import("tensorrt");
  py::class_<CPyProfiler, std::shared_ptr<CPyProfiler> ,nvinfer1::IProfiler>(m, "CPyProfiler")
   //py::class_<PyProfiler,nvinfer1::IProfiler>(m, "PyProfiler")
      .def(py::init<>())
        // The destroy_plugin function does not override the base class, so we must bind it explicitly.
      .def("printLayerTimes", &CPyProfiler::printLayerTimes);
      //.def(py::init<PyProfiler::reportLayerTime_t>());
}

