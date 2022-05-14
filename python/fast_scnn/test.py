import onnx
import onnx_graphsurgeon as gs
graph = gs.import_onnx(onnx.load("/home/nobuo/WorkSpace/jetson/tensorrt-examples/models/fast_scnn_576x576.onnx"))
print([node for node in graph.nodes if node.op == "Resize"][-1].outputs)
