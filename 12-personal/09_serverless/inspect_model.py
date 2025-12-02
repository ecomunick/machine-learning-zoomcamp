# inspect_model.py
import onnx

MODEL_FILE = "hair_classifier_v1.onnx"  # put the model in the same dir

def main():
    model = onnx.load(MODEL_FILE)
    graph = model.graph

    input_names = [i.name for i in graph.input]
    output_names = [o.name for o in graph.output]

    print("INPUT NODES:")
    for n in input_names:
        print(" ", n)
    print("OUTPUT NODES:")
    for n in output_names:
        print(" ", n)

if __name__ == "__main__":
    main()
