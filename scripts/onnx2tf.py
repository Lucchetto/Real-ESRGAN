import argparse
import onnx
from onnx_tf.backend import prepare

def main(args):
    tf_rep = prepare(onnx.load(args.input))
    tf_rep.export_graph(args.output)

if __name__ == '__main__':
    """Convert ONNX model to TensorFlow model"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, help='Input ONNX model path')
    parser.add_argument('--output', type=str, help='Output TensorFlow model path')
    args = parser.parse_args()

    main(args)
