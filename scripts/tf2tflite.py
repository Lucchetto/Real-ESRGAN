import argparse
import tensorflow

def main(args):
    converter = tensorflow.lite.TFLiteConverter.from_saved_model(args.input)
    tflite_model = converter.convert()
    # Save the model.
    with open(args.output, 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    """Convert TensorFlow model to TensorFlow Lite model"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, help='Input TensorFlow model path')
    parser.add_argument('--output', type=str, help='Output TensorFlow Lite model path')
    args = parser.parse_args()

    main(args)
